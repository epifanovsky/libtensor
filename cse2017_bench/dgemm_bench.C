#include <cstdlib>
#include <iostream>
#include <mkl.h>
#include <libutil/timings/timer.h>

void warmup() {

    double a[128*128], b[128*128], c[128*128];
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 128, 128, 128,
        1.0, a, 128, b, 128, 1.0, c, 128);
}


int run_bench(size_t n, unsigned nthr) {

    std::cout << "run_bench(" << n << ", " << nthr << ")" << std::endl;
    size_t n2 = n*n, n4 = n2*n2;
    double *a = (double*)mkl_malloc(sizeof(double)*n4, 64);
    double *b = (double*)mkl_malloc(sizeof(double)*n4, 64);
    double *c = (double*)mkl_malloc(sizeof(double)*n4, 64);
    if(a == 0 || b == 0 || c == 0) {
        std::cout << "mkl_alloc() problem" << std::endl;
        return -1;
    }
    for(size_t i = 0; i < n4; i++) {
        a[i] = double(i+1); b[i] = double(1-i); c[i] = 0.0;
    }
    mkl_set_num_threads(nthr);

    libutil::timer tim;
    tim.start();
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n2, n2, n2,
        3.0, a, n2, b, n2, 2.0, c, n2);
    mkl_free(a); mkl_free(b); mkl_free(c);
    tim.stop();
    std::cout << "dgemm_bench: " << tim.duration() << std::endl;

    std::cout << "SUCCESS" << std::endl;
    return 0;
}


int main(int argc, char **argv) {

    if(argc != 3) {
        std::cout << "Use: \"dgemm_bench N T\", where N is matrix size, "
                     "T is number of threads" << std::endl;
        return -1;
    }

    warmup();

    int n = atoi(argv[1]);
    int t = atoi(argv[2]);
    return run_bench(n, t);
}

