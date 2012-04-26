#ifndef LIBTENSOR_LINALG_ADAPTIVE_IJ_IPQ_JQP_X_H
#define LIBTENSOR_LINALG_ADAPTIVE_IJ_IPQ_JQP_X_H

namespace libtensor {


template<typename M, typename L1, typename L2, typename L3>
void linalg_base_level4_adaptive<M, L1, L2, L3>::ij_ipq_jqp_x(
    size_t ni, size_t nj, size_t np, size_t nq,
    const double *a, size_t spa, size_t sia,
    const double *b, size_t sqb, size_t sjb,
    double *c, size_t sic,
    double d) {

    size_t npq = np * nq;
    size_t npq1 = (npq % 4 == 0) ? npq : npq + 4 - npq % 4;

    bool pqa = (spa == nq), pqb = (sqb == np);

    const double *a1 = 0, *b1 = 0;
    double *a2 = 0, *b2 = 0;
    size_t npqa, npqb;
    bool freea = false, freeb = false;
    bool matrixmultiply = true;

    if(pqa && pqb) {

        if(ni < nj) {

            a2 = M::allocate(ni * npq1);
            freea = true;

            //  a1_iqp <- a_ipq
            for(size_t i = 0; i < ni; i++) {
                L2::ij_ji(nq, np, a + sia * i, spa,
                    a2 + npq1 * i, np);
            }

            a1 = a2; b1 = b; npqa = npq1; npqb = sjb;

        } else {

            b2 = M::allocate(nj * npq1);
            freeb = true;

            //  b1_jpq <- b_jqp
            for(size_t j = 0; j < nj; j++) {
                L2::ij_ji(np, nq, b + sjb * j, sqb,
                    b2 + npq1 * j, nq);
            }

            a1 = a; b1 = b2; npqa = sia; npqb = npq1;

        }

    } else {

        if(pqa) {

            b2 = M::allocate(nj * npq1);
            freeb = true;

            //  b1_jpq <- b_jqp
            for(size_t j = 0; j < nj; j++) {
                L2::ij_ji(np, nq, b + sjb * j, sqb,
                    b2 + npq1 * j, nq);
            }

            a1 = a; b1 = b2; npqa = sia; npqb = npq1;

        } else if(pqb) {

            a2 = M::allocate(ni * npq1);
            freea = true;

            //  a1_iqp <- a_ipq
            for(size_t i = 0; i < ni; i++) {
                L2::ij_ji(nq, np, a + sia * i, spa,
                    a2 + npq1 * i, np);
            }

            a1 = a2; b1 = b; npqa = npq1; npqb = sjb;

        } else {

            matrixmultiply = false;

        }
    }

    if(matrixmultiply) {
        //  c_ij += d * a1_iqp b_jqp
        //  or
        //  c_ij += d * a_ipq b1_jpq
        L3::ij_ip_jp_x(ni, nj, npq, a1, npqa, b1, npqb, c, sic, d);
    } else {
        for(size_t i = 0; i < ni; i++) {
            double *c1 = c + i * sic;
            for(size_t j = 0; j < nj; j++) {
                c1[j] += d * L2::x_pq_qp(np, nq, a + i * sia,
                    spa, b + j * sjb, sqb);
            }
        }
    }

    if(freea) M::deallocate(a2);
    if(freeb) M::deallocate(b2);
}


} // namespace libtensor

#endif // LIBTENSOR_LINALG_ADAPTIVE_IJ_IPQ_JQP_X_H
