#ifndef LIBTENSOR_CTF_TOD_CONTRACT2_IMPL_H
#define LIBTENSOR_CTF_TOD_CONTRACT2_IMPL_H

#include <libtensor/core/bad_dimensions.h>
#include <libtensor/core/mask.h>
#include "../ctf_dense_tensor_ctrl.h"
#include "../ctf_error.h"
#include "../ctf_tod_contract2.h"
#include "ctf_dense_tensor_impl.h"
#include "ctf_tod_copy_impl.h"

namespace libtensor {


template<size_t N, size_t M, size_t K>
const char ctf_tod_contract2<N, M, K>::k_clazz[] =
    "ctf_tod_contract2<N, M, K>";


template<size_t N, size_t M, size_t K>
ctf_tod_contract2<N, M, K>::ctf_tod_contract2(
    const contraction2<N, M, K> &contr,
    ctf_dense_tensor_i<NA, double> &ta,
    ctf_dense_tensor_i<NB, double> &tb,
    double d) :

    m_contr(contr), m_ta(ta), m_tb(tb),
    m_dimsc(contr, ta.get_dims(), tb.get_dims()), m_d(d) {

}


namespace {

template<size_t N, size_t M, size_t K>
ctf_symmetry<N + M, double> ctf_tod_contract2_symmetry_gen(
    const contraction2<N, M, K> &contr,
    const ctf_symmetry<N + K, double> &syma,
    const ctf_symmetry<M + K, double> &symb) {

    enum {
        NA = N + K, NB = M + K, NC = N + M
    };

    const sequence<NA + NB + NC, size_t> &conn = contr.get_conn();
    const sequence<NA, unsigned> &grpa = syma.get_grp();
    const sequence<NA, unsigned> &taga = syma.get_sym();
    const sequence<NB, unsigned> &grpb = symb.get_grp();
    const sequence<NB, unsigned> &tagb = symb.get_sym();

    mask<NA> mappeda;
    mask<NB> mappedb;
    sequence<NA, size_t> mapa;
    sequence<NB, size_t> mapb;

    sequence<NC, unsigned> grpc, tagc;
    unsigned igrp = 0;
    for(size_t i = 0; i < NA; i++) if(conn[NC + i] < NC) {
        if(!mappeda[i]) {
            for(size_t j = i; j < NA; j++) if(grpa[i] == grpa[j]) {
                mapa[grpa[j]] = igrp;
                mappeda[j] = true;
            }
            tagc[igrp] = taga[grpa[i]];
            igrp++;
        }
        grpc[conn[NC + i]] = mapa[grpa[i]];
    }
    for(size_t i = 0; i < NB; i++) if(conn[NC + NA + i] < NC) {
        if(!mappedb[i]) {
            for(size_t j = i; j < NB; j++) if(grpb[i] == grpb[j]) {
                mapb[grpb[j]] = igrp;
                mappedb[j] = true;
            }
            tagc[igrp] = tagb[grpb[i]];
            igrp++;
        }
        grpc[conn[NC + NA + i]] = mapb[grpb[i]];
    }

    return ctf_symmetry<NC, double>(grpc, tagc);
}

template<size_t N, size_t M, size_t K>
ctf_symmetry<N + M, double> ctf_tod_contract2_symmetry(
    const contraction2<N, M, K> &contr,
    const ctf_symmetry<N + K, double> &syma,
    const ctf_symmetry<M + K, double> &symb) {

    return ctf_tod_contract2_symmetry_gen(contr, syma, symb);
}

ctf_symmetry<4, double> ctf_tod_contract2_symmetry(
    const contraction2<2, 2, 2> &contr,
    const ctf_symmetry<4, double> &syma,
    const ctf_symmetry<4, double> &symb) {

    enum {
        NA = 4, NB = 4, NC = 4
    };

    const sequence<NA + NB + NC, size_t> &conn = contr.get_conn();

    bool jilk = false;
    if(syma.is_jilk() && symb.is_jilk()) {
        mask<4> mca, mcb, mcc, m1100, m0011;
        m1100[0] = true; m1100[1] = true; m0011[2] = true; m0011[3] = true;
        for(unsigned i = 0; i < NA; i++) mca[i] = (conn[NC + i] < NC);
        for(unsigned i = 0; i < NB; i++) mcb[i] = (conn[NC + NA + i] < NC);
        for(unsigned i = 0; i < NC; i++) mcc[i] = (conn[i] < NC + NA);
        bool oka = (mca.equals(m1100) || mca.equals(m0011));
        bool okb = (mcb.equals(m1100) || mcb.equals(m0011));
        bool okc = (mcc.equals(m1100) || mcc.equals(m0011));
        if(oka && okb && okc) jilk = true;
    }
    if(jilk) {
        sequence<4, unsigned> grpc, tagc(0);
        for(unsigned i = 0; i < 4; i++) grpc[i] = i;
        return ctf_symmetry<4, double>(grpc, tagc, true);
    }
    return ctf_tod_contract2_symmetry_gen(contr, syma, symb);
}

template<size_t N, size_t M, size_t K>
void ctf_tod_contract2_special(
    const contraction2<N, M, K> &contr,
    const char *map,
    ctf_dense_tensor_i<N + K, double> &ta,
    ctf_dense_tensor_i<M + K, double> &tb,
    double d, bool zero, ctf_dense_tensor_i<N + M, double> &tc) {

    enum {
        NA = N + K, NB = M + K, NC = N + M
    };
}

void ctf_tod_contract2_special(
    const contraction2<2, 2, 2> &contr,
    const char *map,
    ctf_dense_tensor_i<4, double> &ta,
    ctf_dense_tensor_i<4, double> &tb,
    double d, bool zero, ctf_dense_tensor_i<4, double> &tc) {

    enum {
        N = 2, M = 2, K = 2,
        NA = N + K, NB = M + K, NC = N + M
    };

    sequence<NA, unsigned> grpa, taga;
    grpa[0] = 0; grpa[1] = 0; grpa[2] = 1; grpa[3] = 1;
    taga[0] = 0; taga[1] = 0;
    ctf_symmetry<NA, double> syma(grpa, taga);
    taga[0] = 1; taga[1] = 1;
    ctf_symmetry<NA, double> asyma(grpa, taga);

    sequence<NB, unsigned> grpb, tagb;
    grpb[0] = 0; grpb[1] = 0; grpb[2] = 1; grpb[3] = 1;
    tagb[0] = 0; tagb[1] = 0;
    ctf_symmetry<NB, double> symb(grpb, tagb);
    tagb[0] = 1; tagb[1] = 1;
    ctf_symmetry<NB, double> asymb(grpb, tagb);

    sequence<NC, unsigned> grpc, tagc;
    grpc[0] = 0; grpc[1] = 0; grpc[2] = 1; grpc[3] = 1;
    tagc[0] = 0; tagc[1] = 0;
    ctf_symmetry<NC, double> symc(grpc, tagc);
    tagc[0] = 1; tagc[1] = 1;
    ctf_symmetry<NC, double> asymc(grpc, tagc);

    ctf_dense_tensor<NA, double> xa1(ta.get_dims(), syma);
    ctf_dense_tensor<NA, double> xa2(ta.get_dims(), asyma);
    ctf_dense_tensor<NB, double> xb1(tb.get_dims(), symb);
    ctf_dense_tensor<NB, double> xb2(tb.get_dims(), asymb);
    ctf_dense_tensor<NC, double> xc1(tc.get_dims(), symc);
    ctf_dense_tensor<NC, double> xc2(tc.get_dims(), asymc);

    ctf_tod_copy<NA>(ta).perform(true, xa1);
    ctf_tod_copy<NB>(tb).perform(true, xb1);
    ctf_tod_copy<NA>(ta).perform(true, xa2);
    ctf_tod_copy<NB>(tb).perform(true, xb2);

    ctf_tod_contract2<N, M, K>(contr, xa1, xb1).perform(true, xc1);
    ctf_tod_contract2<N, M, K>(contr, xa2, xb2).perform(true, xc2);
    ctf_tod_copy<NC>(xc1, d).perform(zero, tc);
    ctf_tod_copy<NC>(xc2, d).perform(false, tc);

#if 0
    ctf_dense_tensor_ctrl<NA, double> ca(ta);
    ctf_dense_tensor_ctrl<NB, double> cb(tb);

    int symSN[NA] = { SY, NS, NS, NS };
    int symSS[NA] = { SY, NS, SY, NS };
    int symSA[NA] = { SY, NS, AS, NS };
    int symAA[NA] = { AS, NS, AS, NS };

    tCTF_Tensor<double> txa(ca.req_ctf_tensor(), symSN);
    tCTF_Tensor<double> txa1(4, txa.lens, symSS, *txa.wrld);
    tCTF_Tensor<double> txa2t(4, txa.lens, symSA, *txa.wrld);
    txa1["ijkl"] = 0.5 * txa["ijkl"];
    txa2t["ijkl"] = 0.5 * txa["ijkl"];
    tCTF_Tensor<double> txa2(txa2t, symAA);

    tCTF_Tensor<double> txb(cb.req_ctf_tensor(), symSN);
    tCTF_Tensor<double> txb1(4, txb.lens, symSS, *txb.wrld);
    tCTF_Tensor<double> txb2t(4, txb.lens, symSA, *txb.wrld);
    txb1["ijkl"] = 0.5 * txb["ijkl"];
    txb2t["ijkl"] = 0.5 * txb["ijkl"];
    tCTF_Tensor<double> txb2(txb2t, symAA);

    ctf_dense_tensor<NC, double> xc1(tc.get_dims(), symc);
    ctf_dense_tensor<NC, double> xc2(tc.get_dims(), asymc);
    ctf_dense_tensor_ctrl<NC, double> cc1(xc1);
    ctf_dense_tensor_ctrl<NC, double> cc2(xc2);

    cc1.req_ctf_tensor().contract(d, txa1, &map[NC], txb1, &map[NC + NA],
        0.0, &map[0]);
    cc2.req_ctf_tensor().contract(d, txa2, &map[NC], txb2, &map[NC + NA],
        0.0, &map[0]);
    ctf_tod_copy<NC>(xc1).perform(zero, tc);
    ctf_tod_copy<NC>(xc2).perform(false, tc);
#endif
}

} // unnamed namespace

template<size_t N, size_t M, size_t K>
void ctf_tod_contract2<N, M, K>::perform(
    bool zero,
    ctf_dense_tensor_i<NC, double> &tc) {

    static const char method[] =
        "perform(bool, ctf_dense_tensor_i<NC, double>&)";

    if(!m_dimsc.get_dims().equals(tc.get_dims())) {
        throw bad_dimensions(g_ns, k_clazz, method, __FILE__, __LINE__, "tc");
    }

    ctf_dense_tensor_ctrl<NA, double> ca(m_ta);
    ctf_dense_tensor_ctrl<NB, double> cb(m_tb);
    ctf_dense_tensor_ctrl<NC, double> cc(tc);

    ctf_symmetry<NC, double> symc = ctf_tod_contract2_symmetry(m_contr,
        ca.req_symmetry(), cb.req_symmetry());

    char map[NC + NA + NB];
    const sequence<NA + NB + NC, size_t> &conn = m_contr.get_conn();
    for(size_t i = 0; i < NC; i++) {
        map[i] = i + 1;
        size_t ii = conn[NC - i - 1] - NC;
        if(ii < NA) {
            ii = NA - ii - 1;
            map[NC + ii] = i + 1;
        } else {
            ii = NB + NA - ii - 1;
            map[NC + NA + ii] = i + 1;
        }
    }
    for(size_t i = 0, j = NC; i < NA; i++) {
        size_t ii = NC + i;
        if(conn[ii] >= NC + NA) {
            size_t ia = NA - i - 1;
            size_t ib = NC + NA + NB - conn[ii] - 1;
            map[NC + ia] = j + 1;
            map[NC + NA + ib] = j + 1;
            j++;
        }
    }

    if(symc.is_jilk()) {
        ctf_tod_contract2_special(m_contr, map, m_ta, m_tb, m_d, zero, tc);
        return;
    }

    tCTF_Tensor<double> &dta = ca.req_ctf_tensor();
    tCTF_Tensor<double> &dtb = cb.req_ctf_tensor();
    tCTF_Tensor<double> &dtc = cc.req_ctf_tensor();

    double z = ctf_symmetry<NC, double>::symconv_factor(symc,
        cc.req_symmetry());
    dtc.contract(m_d * z, dta, &map[NC], dtb, &map[NC + NA], zero ? 0.0 : 1.0,
        &map[0]);
}


} // namespace libtensor

#endif // LIBTENSOR_CTF_TOD_CONTRACT2_IMPL_H

