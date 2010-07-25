#include <xmmintrin.h>
#include "blas.h"

namespace libtensor {


double blas::ddot_trp(const double *a, const double *b, size_t ni, size_t nj,
	size_t lda, size_t ldb) {

	double d = 0.0;

	bool aligneda = (((size_t)a & 0xF) == 0 && lda % 2 == 0);
	bool alignedb = (((size_t)b & 0xF) == 0 && ldb % 2 == 0);

	if(aligneda && alignedb) {

		size_t mi = 2 * (ni / 2);
		size_t mj = 2 * (nj / 2);

		__m128d r0;
		r0 = _mm_setzero_pd();

		register size_t pq = 0;
		register size_t i = 0;
		for(; i < mi; i += 2) {

			register size_t j = 0;
			register size_t qp = 0;

			for(; j < mj; j += 2) {

				__m128d r1, r2, r3, r4, r5, r6;

				register const double *ptra = a + pq + j,
					*ptrb = b + qp + i;

				r1 = _mm_load_pd(ptra);
				r2 = _mm_load_pd(ptra + lda);
				r3 = _mm_load_pd(ptrb);
				r4 = _mm_load_pd(ptrb + ldb);
				r5 = _mm_shuffle_pd(r3, r4, _MM_SHUFFLE2(0, 0));
				r6 = _mm_shuffle_pd(r3, r4, _MM_SHUFFLE2(1, 1));
				r5 = _mm_mul_pd(r1, r5);
				r6 = _mm_mul_pd(r2, r6);
				r0 = _mm_add_pd(r0, r5);
				r0 = _mm_add_pd(r0, r6);

				qp += 2 * ldb;
			}
			for(; j < nj; j++) {
				d += a[pq +       j] * b[qp + i    ];
				d += a[pq + lda + j] * b[qp + i + 1];
				qp += ldb;
			}
			pq += 2 * lda;
		}
		for(; i < ni; i++) {
			register size_t qp = 0;
			for(register size_t j = 0; j < nj; j++) {
				d += a[pq + j] * b[j * ldb + i];
			}
			pq += lda;
		}

		double dd[] = { 0.0, 0.0 };
		_mm_storeu_pd(dd, r0);

		return d + dd[0] + dd[1];

	} else {

		size_t mj = 4 * (nj / 4);

		register size_t pq = 0;
		for(size_t i = 0; i < ni; i++) {
			register size_t j = 0;
			register size_t qp = 0;
			for(; j < mj; j += 4) {
				d += a[pq + j    ] * b[qp           + i];
				d += a[pq + j + 1] * b[qp +     ldb + i];
				d += a[pq + j + 2] * b[qp + 2 * ldb + i];
				d += a[pq + j + 3] * b[qp + 3 * ldb + i];
				qp += 4 * ldb;
			}
			for(; j < nj; j++) {
				d += a[pq + j] * b[qp + i];
				qp += ldb;
			}
			pq += lda;
		}

		return d;
	}
}


void blas::daxpby_trp(const double *a, double *b, size_t ni, size_t nj,
	size_t si, size_t sj, double ca, double cb) {

	if(cb == 1.0) {
		if(ca == 1.0) {
			blas::daxpby_trp_pp(a, b, ni, nj, si, sj);
		} else if(ca == -1.0) {
			blas::daxpby_trp_pm(a, b, ni, nj, si, sj);
		} else {
			blas::daxpby_trp_pa(a, b, ni, nj, si, sj, ca);
		}
	} else if(cb == 0.0) {
		if(ca == 1.0) {
			blas::daxpby_trp_zp(a, b, ni, nj, si, sj);
		} else {
			blas::daxpby_trp_za(a, b, ni, nj, si, sj, ca);
		}
	} else {
		blas::daxpby_trp_aa(a, b, ni, nj, si, sj, ca, cb);
	}
}

void blas::daxpby_trp_aa(const double *a, double *b, size_t ni, size_t nj,
	size_t si, size_t sj, double ca, double cb) {

	size_t i = 0, j = 0;
	size_t mi = 4*(ni/4), mj = 4*(nj/4);

	for(i = 0; i < mi; i += 4) {

		register size_t ij = i * sj;
		register size_t ji = i;

		for (j = 0; j < mj; j += 4) {

			b[ij  ] = cb*b[ij  ] + ca*a[ji     ];
			b[ij+1] = cb*b[ij+1] + ca*a[ji+  si];
			b[ij+2] = cb*b[ij+2] + ca*a[ji+2*si];
			b[ij+3] = cb*b[ij+3] + ca*a[ji+3*si];

			b[ij+sj  ] = cb*b[ij+sj  ] + ca*a[ji     +1];
			b[ij+sj+1] = cb*b[ij+sj+1] + ca*a[ji+  si+1];
			b[ij+sj+2] = cb*b[ij+sj+2] + ca*a[ji+2*si+1];
			b[ij+sj+3] = cb*b[ij+sj+3] + ca*a[ji+3*si+1];

			b[ij+2*sj  ] = cb*b[ij+2*sj  ] + ca*a[ji     +2];
			b[ij+2*sj+1] = cb*b[ij+2*sj+1] + ca*a[ji+  si+2];
			b[ij+2*sj+2] = cb*b[ij+2*sj+2] + ca*a[ji+2*si+2];
			b[ij+2*sj+3] = cb*b[ij+2*sj+3] + ca*a[ji+3*si+2];

			b[ij+3*sj  ] = cb*b[ij+3*sj  ] + ca*a[ji     +3];
			b[ij+3*sj+1] = cb*b[ij+3*sj+1] + ca*a[ji+  si+3];
			b[ij+3*sj+2] = cb*b[ij+3*sj+2] + ca*a[ji+2*si+3];
			b[ij+3*sj+3] = cb*b[ij+3*sj+3] + ca*a[ji+3*si+3];

			ij += 4;
			ji += 4*si;
		}

		for (; j < nj; j++) {

			b[ij     ] = cb*b[ij     ] + ca*a[ji  ];
			b[ij+  sj] = cb*b[ij+  sj] + ca*a[ji+1];
			b[ij+2*sj] = cb*b[ij+2*sj] + ca*a[ji+2];
			b[ij+3*sj] = cb*b[ij+3*sj] + ca*a[ji+3];

			ij++;
			ji += si;
		}
	}

	for (; i < ni; i++) {

		register size_t ij = i * sj;
		register size_t ji = i;

		for (j = 0; j < mj; j += 4) {

			b[ij  ] = cb*b[ij  ] + ca*a[ji     ];
			b[ij+1] = cb*b[ij+1] + ca*a[ji+  si];
			b[ij+2] = cb*b[ij+2] + ca*a[ji+2*si];
			b[ij+3] = cb*b[ij+3] + ca*a[ji+3*si];

			ij += 4;
			ji += 4*si;
		}

		for (; j < nj; j++) {

			b[ij] = cb*b[ij] + ca*a[ji];

			ij++;
			ji += si;
		}
	}
}

void blas::daxpby_trp_pa(const double *a, double *b, size_t ni, size_t nj,
	size_t si, size_t sj, double ca) {

	size_t i = 0, j = 0;
	size_t mi = 4*(ni/4), mj = 4*(nj/4);

	for(i = 0; i < mi; i += 4) {

		register size_t ij = i * sj;
		register size_t ji = i;

		for (j = 0; j < mj; j += 4) {

			b[ij  ] += ca*a[ji     ];
			b[ij+1] += ca*a[ji+  si];
			b[ij+2] += ca*a[ji+2*si];
			b[ij+3] += ca*a[ji+3*si];

			b[ij+sj  ] += ca*a[ji     +1];
			b[ij+sj+1] += ca*a[ji+  si+1];
			b[ij+sj+2] += ca*a[ji+2*si+1];
			b[ij+sj+3] += ca*a[ji+3*si+1];

			b[ij+2*sj  ] += ca*a[ji     +2];
			b[ij+2*sj+1] += ca*a[ji+  si+2];
			b[ij+2*sj+2] += ca*a[ji+2*si+2];
			b[ij+2*sj+3] += ca*a[ji+3*si+2];

			b[ij+3*sj  ] += ca*a[ji     +3];
			b[ij+3*sj+1] += ca*a[ji+  si+3];
			b[ij+3*sj+2] += ca*a[ji+2*si+3];
			b[ij+3*sj+3] += ca*a[ji+3*si+3];

			ij += 4;
			ji += 4*si;
		}

		for (; j < nj; j++) {

			b[ij     ] += ca*a[ji  ];
			b[ij+  sj] += ca*a[ji+1];
			b[ij+2*sj] += ca*a[ji+2];
			b[ij+3*sj] += ca*a[ji+3];

			ij++;
			ji += si;
		}
	}

	for (; i < ni; i++) {

		register size_t ij = i * sj;
		register size_t ji = i;

		for (j = 0; j < mj; j += 4) {

			b[ij  ] += ca*a[ji     ];
			b[ij+1] += ca*a[ji+  si];
			b[ij+2] += ca*a[ji+2*si];
			b[ij+3] += ca*a[ji+3*si];

			ij += 4;
			ji += 4*si;
		}

		for (; j < nj; j++) {

			b[ij] += ca*a[ji];

			ij++;
			ji += si;
		}
	}
}

void blas::daxpby_trp_pp(const double *a, double *b, size_t ni, size_t nj,
	size_t si, size_t sj) {

	size_t i = 0, j = 0;
	size_t mi = 4*(ni/4), mj = 4*(nj/4);

	for(i = 0; i < mi; i += 4) {

		register size_t ij = i * sj;
		register size_t ji = i;

		for (j = 0; j < mj; j += 4) {

			b[ij  ] += a[ji     ];
			b[ij+1] += a[ji+  si];
			b[ij+2] += a[ji+2*si];
			b[ij+3] += a[ji+3*si];

			b[ij+sj  ] += a[ji     +1];
			b[ij+sj+1] += a[ji+  si+1];
			b[ij+sj+2] += a[ji+2*si+1];
			b[ij+sj+3] += a[ji+3*si+1];

			b[ij+2*sj  ] += a[ji     +2];
			b[ij+2*sj+1] += a[ji+  si+2];
			b[ij+2*sj+2] += a[ji+2*si+2];
			b[ij+2*sj+3] += a[ji+3*si+2];

			b[ij+3*sj  ] += a[ji     +3];
			b[ij+3*sj+1] += a[ji+  si+3];
			b[ij+3*sj+2] += a[ji+2*si+3];
			b[ij+3*sj+3] += a[ji+3*si+3];

			ij += 4;
			ji += 4*si;
		}

		for (; j < nj; j++) {

			b[ij     ] += a[ji  ];
			b[ij+  sj] += a[ji+1];
			b[ij+2*sj] += a[ji+2];
			b[ij+3*sj] += a[ji+3];

			ij++;
			ji += si;
		}
	}

	for (; i < ni; i++) {

		register size_t ij = i * sj;
		register size_t ji = i;

		for (j = 0; j < mj; j += 4) {

			b[ij  ] += a[ji     ];
			b[ij+1] += a[ji+  si];
			b[ij+2] += a[ji+2*si];
			b[ij+3] += a[ji+3*si];

			ij += 4;
			ji += 4*si;
		}

		for (; j < nj; j++) {

			b[ij] += a[ji];

			ij++;
			ji += si;
		}
	}
}

void blas::daxpby_trp_pm(const double *a, double *b, size_t ni, size_t nj,
	size_t si, size_t sj) {

	size_t i = 0, j = 0;
	size_t mi = 4*(ni/4), mj = 4*(nj/4);

	for(i = 0; i < mi; i += 4) {

		register size_t ij = i * sj;
		register size_t ji = i;

		for (j = 0; j < mj; j += 4) {

			b[ij  ] -= a[ji     ];
			b[ij+1] -= a[ji+  si];
			b[ij+2] -= a[ji+2*si];
			b[ij+3] -= a[ji+3*si];

			b[ij+sj  ] -= a[ji     +1];
			b[ij+sj+1] -= a[ji+  si+1];
			b[ij+sj+2] -= a[ji+2*si+1];
			b[ij+sj+3] -= a[ji+3*si+1];

			b[ij+2*sj  ] -= a[ji     +2];
			b[ij+2*sj+1] -= a[ji+  si+2];
			b[ij+2*sj+2] -= a[ji+2*si+2];
			b[ij+2*sj+3] -= a[ji+3*si+2];

			b[ij+3*sj  ] -= a[ji     +3];
			b[ij+3*sj+1] -= a[ji+  si+3];
			b[ij+3*sj+2] -= a[ji+2*si+3];
			b[ij+3*sj+3] -= a[ji+3*si+3];

			ij += 4;
			ji += 4*si;
		}

		for (; j < nj; j++) {

			b[ij     ] -= a[ji  ];
			b[ij+  sj] -= a[ji+1];
			b[ij+2*sj] -= a[ji+2];
			b[ij+3*sj] -= a[ji+3];

			ij++;
			ji += si;
		}
	}

	for (; i < ni; i++) {

		register size_t ij = i * sj;
		register size_t ji = i;

		for (j = 0; j < mj; j += 4) {

			b[ij  ] -= a[ji     ];
			b[ij+1] -= a[ji+  si];
			b[ij+2] -= a[ji+2*si];
			b[ij+3] -= a[ji+3*si];

			ij += 4;
			ji += 4*si;
		}

		for (; j < nj; j++) {

			b[ij] -= a[ji];

			ij++;
			ji += si;
		}
	}
}

void blas::daxpby_trp_za(const double *a, double *b, size_t ni, size_t nj,
	size_t si, size_t sj, double ca) {

	size_t i = 0, j = 0;
	size_t mi = 4*(ni/4), mj = 4*(nj/4);

	for(i = 0; i < mi; i += 4) {

		register size_t ij = i * sj;
		register size_t ji = i;

		for (j = 0; j < mj; j += 4) {

			b[ij  ] = ca*a[ji     ];
			b[ij+1] = ca*a[ji+  si];
			b[ij+2] = ca*a[ji+2*si];
			b[ij+3] = ca*a[ji+3*si];

			b[ij+sj  ] = ca*a[ji     +1];
			b[ij+sj+1] = ca*a[ji+  si+1];
			b[ij+sj+2] = ca*a[ji+2*si+1];
			b[ij+sj+3] = ca*a[ji+3*si+1];

			b[ij+2*sj  ] = ca*a[ji     +2];
			b[ij+2*sj+1] = ca*a[ji+  si+2];
			b[ij+2*sj+2] = ca*a[ji+2*si+2];
			b[ij+2*sj+3] = ca*a[ji+3*si+2];

			b[ij+3*sj  ] = ca*a[ji     +3];
			b[ij+3*sj+1] = ca*a[ji+  si+3];
			b[ij+3*sj+2] = ca*a[ji+2*si+3];
			b[ij+3*sj+3] = ca*a[ji+3*si+3];

			ij += 4;
			ji += 4*si;
		}

		for (; j < nj; j++) {

			b[ij     ] = ca*a[ji  ];
			b[ij+  sj] = ca*a[ji+1];
			b[ij+2*sj] = ca*a[ji+2];
			b[ij+3*sj] = ca*a[ji+3];

			ij++;
			ji += si;
		}
	}

	for (; i < ni; i++) {

		register size_t ij = i * sj;
		register size_t ji = i;

		for (j = 0; j < mj; j += 4) {

			b[ij  ] = ca*a[ji     ];
			b[ij+1] = ca*a[ji+  si];
			b[ij+2] = ca*a[ji+2*si];
			b[ij+3] = ca*a[ji+3*si];

			ij += 4;
			ji += 4*si;
		}

		for (; j < nj; j++) {

			b[ij] = ca*a[ji];

			ij++;
			ji += si;
		}
	}
}

void blas::daxpby_trp_zp(const double *a, double *b, size_t ni, size_t nj,
	size_t si, size_t sj) {

	size_t i = 0, j = 0;
	size_t mi = 4*(ni/4), mj = 4*(nj/4);

	for(i = 0; i < mi; i += 4) {

		register size_t ij = i * sj;
		register size_t ji = i;

		for (j = 0; j < mj; j += 4) {

			b[ij  ] = a[ji     ];
			b[ij+1] = a[ji+  si];
			b[ij+2] = a[ji+2*si];
			b[ij+3] = a[ji+3*si];

			b[ij+sj  ] = a[ji     +1];
			b[ij+sj+1] = a[ji+  si+1];
			b[ij+sj+2] = a[ji+2*si+1];
			b[ij+sj+3] = a[ji+3*si+1];

			b[ij+2*sj  ] = a[ji     +2];
			b[ij+2*sj+1] = a[ji+  si+2];
			b[ij+2*sj+2] = a[ji+2*si+2];
			b[ij+2*sj+3] = a[ji+3*si+2];

			b[ij+3*sj  ] = a[ji     +3];
			b[ij+3*sj+1] = a[ji+  si+3];
			b[ij+3*sj+2] = a[ji+2*si+3];
			b[ij+3*sj+3] = a[ji+3*si+3];

			ij += 4;
			ji += 4*si;
		}

		for (; j < nj; j++) {

			b[ij     ] = a[ji  ];
			b[ij+  sj] = a[ji+1];
			b[ij+2*sj] = a[ji+2];
			b[ij+3*sj] = a[ji+3];

			ij++;
			ji += si;
		}
	}

	for (; i < ni; i++) {

		register size_t ij = i * sj;
		register size_t ji = i;

		for (j = 0; j < mj; j += 4) {

			b[ij  ] = a[ji     ];
			b[ij+1] = a[ji+  si];
			b[ij+2] = a[ji+2*si];
			b[ij+3] = a[ji+3*si];

			ij += 4;
			ji += 4*si;
		}

		for (; j < nj; j++) {

			b[ij] = a[ji];

			ij++;
			ji += si;
		}
	}
}

} // namespace libtensor
