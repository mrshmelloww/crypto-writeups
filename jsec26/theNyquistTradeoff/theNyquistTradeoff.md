# JSEC26 \ The Nyquist Tradeoff

 *Recovering an MT19937 Seed via Matrix Discrete Logarithm & BSGS*
## Story Time

We got our hands on the stems for an unreleased Hatsune Miku track (greatest Vocaloid of all time???), but the producer clearly had trust issues, or maybe just severe OCD.

The files attached with the challenge were:
- `dsp_engine.cpp`
- `bounced_audio.log`

The engine basically takes some matrix, applies the same transformation to it over and over, and keeps going for a fixed number of steps. That step count is then used as the seed for a 64-bit Mersenne Twister, and the resulting keystream is XORed with the plaintext to produce the final encrypted output.

At the end, we’re given the final matrix state along with the ciphertext dumped in a log file, so everything boils down to recovering how many steps were applied.

## Understanding the Attack

The `dsp_engine.cpp` reveals that the filter updates its state (`freq_mod` and `phase_shift`) using a linear transformation. In mathematical terms, this is a $1 \times 2$ vector being multiplied by a $2 \times 2$ matrix $M$ modulo a large prime $P = 999999999999999989$.

So to recover $K$ we need to know what exponent ends up yielding the target matrix. Sounds familiar doesn't it? Well that's because it's a **Discrete Logarithm Problem (DLP)**:

$$V_{start} \cdot M^K \equiv V_{target} \pmod P$$

Before attempting to recover the key $K$, we must ensure the matrix transition possesses a long enough cycle. If the matrix has a small order, it could loop prematurely, making it impossible to reach a state at $\approx 7 \times 10^{15}$ steps.

To confirm $M$ behaves as a true generator of this group, we verify:
1. $M^{P-1} \equiv I \pmod P$
2. $M^{(P-1)/q} \not\equiv I \pmod P$ for every prime factor $q$ of $P-1$.

If both conditions hold, the matrix has full order $P-1$ and is guaranteed not to loop early. We can check for that with:
```py
def is_gen(M, P, factors):
    # Flattened 2x2 matrix multiplication
    mul = lambda A, B: [(A[0]*B[0] + A[1]*B[2]) % P, (A[0]*B[1] + A[1]*B[3]) % P, 
                        (A[2]*B[0] + A[3]*B[2]) % P, (A[2]*B[1] + A[3]*B[3]) % P]
    
    # Fast matrix exponentiation
    def mpow(A, e, R=[1, 0, 0, 1]):
        while e: R, A, e = (mul(R, A) if e & 1 else R), mul(A, A), e >> 1
        return R

    I = [1, 0, 0, 1]
    return mpow(M, P - 1) == I and all(mpow(M, (P - 1) // q) != I for q in factors)

# Flattened matrix: [m11, m12, m21, m22]
M = [1337, 80085, 42, 2025]
P = 999999999999999989
factors = [2, 3, 16666667]

print(is_gen(M, P, factors)) #returns TRUE
```

So it's just like checking for the primitive root, except here it's with matrices and their multiplicative neutral counterpart.

Now that we confirmed this is does fullfill the conditions of a DLP, we can begin solving it using the **Baby-Step Giant-Step Algorithm (BSGS)**. 

The algorithm revolves around the logic of rewriting the exponent $k$ as $k = mj + i$. Here, $m$ is ideally set to $\approx \sqrt{\phi(P)}$ to minimize the values of $i$ and $j$, allowing for an $O(\sqrt{P})$ time complexity. 

We are searching for a state collision where:
$$V_{start} \cdot M^i \equiv V_{target} \cdot (M^{-m})^j \pmod P$$

By iterating through generated values, we look for a collision between the correct $(i, j)$ pairs. Since we are solving this in C++, we store the precomputed "baby steps" in a vector. However, given the size of our state space where $\sqrt{P} \approx \sqrt{10^{18}} = 10^9$, we cannot simply allocate a vector of $10^9$ elements, it would require an impossible amount of RAM.

To solve this, we apply the **Division Algorithm**. If $k = qm + r$, we know that $r < m$ is our worst-case remainder. We choose a value for $m$ (the `buffer_size`) that makes allocating a vector feasible for our "baby steps" ($i$). For the quotient ($j$), we leverage the fact that matrix multiplication is associative. This allows us to treat each "giant step" as its own state, jumping backward from the target and searching for a matching baby step in our vector using binary search, after that we can discard the state because we know we won't be using that again. This way we reduce the value of $i$ and increase the value of $j$ whilst still have it execute with reasonable time and memory limits.

This approach effectively trades time for memory. While we can afford to let the code run for several minutes, we cannot reserve an infinite amount of storage. Once the collision between $i$ and $j$ is identified, we can successfully recover the total sample count $K$.

## Implementing the Attack

We can start by defining a few helper functions and a structure to help us implement the solution:

```c++
const long long P = 999999999999999989LL;

struct SynthState { 
    long long freq_mod, phase_shift; 
    int idx;
};

bool compareStates(const SynthState& a, const SynthState& b) {
    if (a.freq_mod != b.freq_mod) return a.freq_mod < b.freq_mod;
    return a.phase_shift < b.phase_shift;
}

SynthState mult(long long A_xx, long long A_xy, long long A_yx, long long A_yy, SynthState v) {
    long long nx = (long long)(((__int128_t)A_xx * v.freq_mod + (__int128_t)A_xy * v.phase_shift) % P);
    long long ny = (long long)(((__int128_t)A_yx * v.freq_mod + (__int128_t)A_yy * v.phase_shift) % P);
    return {nx < 0 ? nx + P : nx, ny < 0 ? ny + P : ny, v.idx};
}

long long power(long long base, long long exp) {
    long long res = 1;
    base %= P;
    while (exp > 0) {
        if (exp % 2 == 1) res = (long long)((__int128_t)res * base % P);
        base = (long long)((__int128_t)base * base % P);
        exp /= 2;
    }
    return res;
}```

With that out of the way, we can now get to our main solver. We start by calculating our baby steps:

```c++
    long long buffer_size = 25000000;
    
    std::cout << "[*] Allocating RAM for 25M cached audio samples..." << std::endl;
    std::vector<SynthState> cached_samples;
    cached_samples.reserve(buffer_size + 1);
    
    SynthState current = {1, 1, 0};
    for(int i = 0; i < buffer_size; i++) {
        cached_samples.push_back(current);
        current = mult(1337, 80085, 42, 2025, current);
        current.idx = i + 1;
    }
    
    std::cout << "[*] Sorting cached waveform buffer..." << std::endl;
    std::sort(cached_samples.begin(), cached_samples.end(), compareStates);
```

Of course, we sort them to allow for binary search later on. Now we calculate the inverse of our matrix so we can use it in the arithmetic of our giant steps:

```c++
    long long det = (1337 * 2025 - 80085 * 42) % P;
    if (det < 0) det += P;
    long long invDet = modInverse(det);
    
    long long invA_xx = (long long)(((__int128_t)2025 * invDet) % P);
    long long invA_xy = (long long)(((__int128_t)-80085 * invDet) % P);
    long long invA_yx = (long long)(((__int128_t)-42 * invDet) % P);
    long long invA_yy = (long long)(((__int128_t)1337 * invDet) % P);
    
    // Ensure absolute positive bounds after the 128-bit modulo
    if (invA_xx < 0) invA_xx += P;
    if (invA_xy < 0) invA_xy += P;
    if (invA_yx < 0) invA_yx += P;
    if (invA_yy < 0) invA_yy += P;

    long long stepA_xx = 1, stepA_xy = 0, stepA_yx = 0, stepA_yy = 1;
    for (int i = 0; i < buffer_size; i++) {
        long long nx = (long long)(((__int128_t)invA_xx * stepA_xx + (__int128_t)invA_xy * stepA_yx) % P);
        long long ny = (long long)(((__int128_t)invA_xx * stepA_xy + (__int128_t)invA_xy * stepA_yy) % P);
        long long mx = (long long)(((__int128_t)invA_yx * stepA_xx + (__int128_t)invA_yy * stepA_yx) % P);
        long long my = (long long)(((__int128_t)invA_yx * stepA_xy + (__int128_t)invA_yy * stepA_yy) % P);
        stepA_xx = nx; stepA_xy = ny; stepA_yx = mx; stepA_yy = my;
    }
```

Now that we have our inverse, we can start caclulating our giant steps:

```c++
SynthState current_g = {909215786004298781LL, 199738637349011533, 0}; 
    long long K = -1;
    
    std::cout << "[*] Fast-forwarding audio blocks (Searching for phase collision)..." << std::endl;
    for(long long j = 0; j <= 312000000; j++) {
        int low = 0, high = cached_samples.size() - 1;
        int match_idx = -1;
        while (low <= high) {
            int mid = low + (high - low) / 2;
            if (cached_samples[mid].freq_mod == current_g.freq_mod && cached_samples[mid].phase_shift == current_g.phase_shift) {
                match_idx = cached_samples[mid].idx;
                break;
            }
            if (cached_samples[mid].freq_mod < current_g.freq_mod ||
                    (cached_samples[mid].freq_mod == current_g.freq_mod 
                    && cached_samples[mid].phase_shift < current_g.phase_shift)) {
                low = mid + 1;
            } else {
                high = mid - 1; 
            }
        }
        
        if (match_idx != -1) {
            K = j * buffer_size + match_idx;
            break;
        }
        current_g = mult(stepA_xx, stepA_xy, stepA_yx, stepA_yy, current_g);
        
        if (j % 10000000 == 0 && j > 0) std::cout << "  -> Filtered " << j << " audio blocks..." << std::endl;
    }
    
    std::cout << "[+] Recovered Target Sample Count (K): " << K << "\n";
```

As you can see, for each giant step we take we perform binary search to check if there exists an answer or not, and we keep looping until we find our answer. We also reserved less memory while looping for longer to make this thing feasible. In the end, we can finally decrypt the ciphertext and recover the plaintext:

```c++
if (K != -1) {
        std::vector<int> ciphertext = {};

        std::mt19937_64 rng(K);
        std::cout << "[-] DECODED_VOCAL_TRACK: ";
        for (int c : ciphertext) {
            std::cout << (char)(c ^ (rng() & 0xFF));
        }
        std::cout << "\n";
    } else {
        std::cout << "[!] Audio sync failed. K not found.\n";
    }
```

this leaves us with the flag `JSEC{m4tr1x_bsgs_fl4tt3ns_un0rd3r3d_m4p}`.

## Final Thoughts
This challenge was intended to demonstrate the intersection of Linear Algebra and Cryptography in a way that punishes a "black box" approach to DSP. While a producer might see these matrix transitions as a complex way to shift audio phases, from an adversarial perspective, it is simply a state machine operating over a finite field.

The core lesson here is the **Reduction of Complexity**. By proving that the matrix is a **valid generator** for the group, we established that the path is unique and non-repeating, but also that it follows the predictable structure of the Discrete Logarithm Problem. This exposed the "7 quadrillion" step count as a deceptive barrier; while it sounds like an insurmountable brute-force target, the group's structure makes it vulnerable to a sub-linear **Meet-in-the-Middle** attack.

Ultimately, the challenge forced a trade-off between memory and time. We couldn't reserve infinite memory for a full $\sqrt{P}$ table, but by using the **Division Algorithm** to pick a feasible $m$, we proved that even massive state spaces can be traversed in minutes with the right mathematical leverage. If you don't understand the group your generator lives in, your "massive" period is just an illusion.

##### [Check out the challenge files](https://github.com/mrshmelloww/crypto-writeups/tree/main/cybereto/theLeakyDevice/challengeFiles)
