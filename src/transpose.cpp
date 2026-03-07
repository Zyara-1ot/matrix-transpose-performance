#include <bits/stdc++.h>
#include<immintrin.h>
#include <sys/mman.h>
using namespace std;

void transpose(const int*, int*, int, int);
void avxtps(const int* poa, int* pob, int poa_stride, int pob_stride);
int main(){
    int m;
    int n;
    n = 4096; m = 4096;
    int pad_m = m + 16;
    int pad_n = n + 16;
    int* A = (int*)mmap(nullptr, (size_t)n * pad_m * sizeof(int),
    PROT_READ|PROT_WRITE,
    MAP_PRIVATE|MAP_ANONYMOUS|MAP_HUGETLB, -1, 0);
assert(A != MAP_FAILED);

   int* B = (int*)mmap(nullptr, (size_t)n * pad_n * sizeof(int),
    PROT_READ|PROT_WRITE,
    MAP_PRIVATE|MAP_ANONYMOUS|MAP_HUGETLB, -1, 0);
assert(B != MAP_FAILED);
    int declr = 100;
    long long count = 0;
    for(int i = 0; i < n; i++){
        for(int j = 0; j < pad_m; j++){
            A[i * pad_m + j] = i * pad_m + j;
        }
    }

    for(int i = 0; i < declr; i++){
        auto start = std::chrono::high_resolution_clock::now();
        transpose(A, B, n, m);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        count += duration.count();
    }
    double avgtym = (double)count / declr;
    double sec = avgtym / 1000.0;
    double bytes = 2 * n * m * sizeof(int);
    double speed = bytes / sec;
    double gbps = speed / 1e9;
    cout << gbps << endl;

    munmap(A, (size_t)n * pad_m * sizeof(int));
    munmap(B, (size_t)n * pad_n * sizeof(int)); 

    return 0;
}

void transpose(const int* A, int* B, int n, int m){
    int t = 64;
    int pad_m = m + 16;
    int pad_n = n + 16;
    //for(int i = 0; i < n; i++){
    //    for(int j = 0; j < m; j++){
       //     B[j * n + i] = A[i * m + j];
       //     
      //  }
    //}
    #pragma omp parallel for collapse(2) schedule(guided)
    for(int i = 0; i < n; i+=8){
        for(int j = 0; j < m; j+=8){
            avxtps(A + i*pad_m + j, B + j*pad_n + i, pad_m, pad_n);
        }
    }
}


void avxtps(const int* a, int* b, int a_pad, int b_pad){

    __m256i r0 = _mm256_loadu_si256((__m256i*)(a + 0 * a_pad));
    __m256i r1 = _mm256_loadu_si256((__m256i*)(a + 1 * a_pad));
    __m256i r2 = _mm256_loadu_si256((__m256i*)(a + 2 * a_pad));
    __m256i r3 = _mm256_loadu_si256((__m256i*)(a + 3 * a_pad));
    __m256i r4 = _mm256_loadu_si256((__m256i*)(a + 4 * a_pad));
    __m256i r5 = _mm256_loadu_si256((__m256i*)(a + 5 * a_pad));
    __m256i r6 = _mm256_loadu_si256((__m256i*)(a + 6 * a_pad));
    __m256i r7 = _mm256_loadu_si256((__m256i*)(a + 7 * a_pad));

    __m256i t0 = _mm256_unpacklo_epi32(r0, r1);
    __m256i t1 = _mm256_unpackhi_epi32(r0, r1);
    __m256i t2 = _mm256_unpacklo_epi32(r2, r3);
    __m256i t3 = _mm256_unpackhi_epi32(r2, r3);
    __m256i t4 = _mm256_unpacklo_epi32(r4, r5);
    __m256i t5 = _mm256_unpackhi_epi32(r4, r5);
    __m256i t6 = _mm256_unpacklo_epi32(r6, r7);
    __m256i t7 = _mm256_unpackhi_epi32(r6, r7);

    __m256i z0 = _mm256_unpacklo_epi64(t0, t2);
    __m256i z1 = _mm256_unpackhi_epi64(t0, t2);
    __m256i z2 = _mm256_unpacklo_epi64(t1, t3);
    __m256i z3 = _mm256_unpackhi_epi64(t1, t3);
    __m256i z4 = _mm256_unpacklo_epi64(t4, t6);
    __m256i z5 = _mm256_unpackhi_epi64(t4, t6);
    __m256i z6 = _mm256_unpacklo_epi64(t5, t7);
    __m256i z7 = _mm256_unpackhi_epi64(t5, t7); 


    __m256i w0 = _mm256_permute2x128_si256(z0, z4, 0x20);
    __m256i w1 = _mm256_permute2x128_si256(z2, z6, 0x20);
    __m256i w2 = _mm256_permute2x128_si256(z1, z5, 0x20);
    __m256i w3 = _mm256_permute2x128_si256(z3, z7, 0x20);
    __m256i w4 = _mm256_permute2x128_si256(z0, z4, 0x31);
    __m256i w5 = _mm256_permute2x128_si256(z2, z6, 0x31);
    __m256i w6 = _mm256_permute2x128_si256(z1, z5, 0x31);
    __m256i w7 = _mm256_permute2x128_si256(z3, z7, 0x31);


    _mm256_storeu_si256((__m256i*)(b + 0 * b_pad), w0);
    _mm256_storeu_si256((__m256i*)(b + 1 * b_pad), w2);
    _mm256_storeu_si256((__m256i*)(b + 2 * b_pad), w1);
    _mm256_storeu_si256((__m256i*)(b + 3 * b_pad), w3);
    _mm256_storeu_si256((__m256i*)(b + 4 * b_pad), w4);
    _mm256_storeu_si256((__m256i*)(b + 5 * b_pad), w6);
    _mm256_storeu_si256((__m256i*)(b + 6 * b_pad), w5);
    _mm256_storeu_si256((__m256i*)(b + 7 * b_pad), w7);








}