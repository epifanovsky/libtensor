#include <libtensor/core/bad_dimensions.h>
#include "../diag_tensor_ctrl.h"
#include "../diag_tod_dotprod.h"
#include "diag_tod_aux_dotprod.h"

namespace libtensor {


template<size_t N>
const char diag_tod_dotprod<N>::k_clazz[] = "diag_tod_dotprod<N>";


template<size_t N>
diag_tod_dotprod<N>::diag_tod_dotprod(
    diag_tensor_rd_i<N, double> &dta,
    diag_tensor_rd_i<N, double> &dtb) :

    m_dta(dta), m_dtb(dtb) {

    static const char method[] = "diag_tod_dotprod("
        "diag_tensor_rd_i<N, double>&, diag_tensor_rd_i<N, double>&)";

    if(!verify_dims()) {
        throw bad_dimensions(g_ns, k_clazz, method, __FILE__, __LINE__,
            "dta,dtb");
    }
}


template<size_t N>
diag_tod_dotprod<N>::diag_tod_dotprod(
    diag_tensor_rd_i<N, double> &dta, const permutation<N> &perma,
    diag_tensor_rd_i<N, double> &dtb, const permutation<N> &permb) :

    m_dta(dta), m_dtb(dtb), m_tra(perma), m_trb(permb) {

    static const char method[] = "diag_tod_dotprod("
        "diag_tensor_rd_i<N, double>&, const permutation<N>&, "
        "diag_tensor_rd_i<N, double>&, const permutation<N>&)";

    if(!verify_dims()) {
        throw bad_dimensions(g_ns, k_clazz, method, __FILE__, __LINE__,
            "dta,dtb");
    }
}


template<size_t N>
diag_tod_dotprod<N>::diag_tod_dotprod(
    diag_tensor_rd_i<N, double> &dta, const tensor_transf<N, double> &tra,
    diag_tensor_rd_i<N, double> &dtb, const tensor_transf<N, double> &trb) :

    m_dta(dta), m_dtb(dtb), m_tra(tra), m_trb(trb) {

    static const char method[] = "diag_tod_dotprod("
        "diag_tensor_rd_i<N, double>&, const tensor_transf<N, double>&, "
        "diag_tensor_rd_i<N, double>&, const tensor_transf<N, double>&)";

    if(!verify_dims()) {
        throw bad_dimensions(g_ns, k_clazz, method, __FILE__, __LINE__,
            "dta,dtb");
    }
}


template<size_t N>
void diag_tod_dotprod<N>::prefetch() {

}


template<size_t N>
double diag_tod_dotprod<N>::calculate() {

    double d = 0.0;

    diag_tod_dotprod::start_timer();

    try {

        diag_tensor_rd_ctrl<N, double> ca(m_dta);
        diag_tensor_rd_ctrl<N, double> cb(m_dtb);

        const diag_tensor_space<N> &dtsa = m_dta.get_space();
        const diag_tensor_space<N> &dtsb = m_dtb.get_space();

        std::vector<size_t> ssla, sslb;
        dtsa.get_all_subspaces(ssla);
        dtsb.get_all_subspaces(sslb);

        for(size_t ssia = 0; ssia < ssla.size(); ssia++) {
        for(size_t ssib = 0; ssib < sslb.size(); ssib++) {

            const diag_tensor_subspace<N> &ssa = dtsa.get_subspace(ssla[ssia]);
            const diag_tensor_subspace<N> &ssb = dtsb.get_subspace(sslb[ssib]);
            size_t sza = dtsa.get_subspace_size(ssla[ssia]);
            size_t szb = dtsb.get_subspace_size(sslb[ssib]);

            const double *pa = ca.req_const_dataptr(ssla[ssia]);
            const double *pb = cb.req_const_dataptr(sslb[ssib]);
            d += diag_tod_aux_dotprod<N>(dtsa.get_dims(), dtsb.get_dims(),
                ssa, ssb, pa, pb, sza, szb, m_tra, m_trb).calculate();
            cb.ret_const_dataptr(sslb[ssib], pb);
            ca.ret_const_dataptr(ssla[ssia], pa);
        }
        }

    } catch(...) {
        diag_tod_dotprod::stop_timer();
        throw;
    }

    diag_tod_dotprod::stop_timer();

    return d;
}


template<size_t N>
bool diag_tod_dotprod<N>::verify_dims() {

    dimensions<N> dimsa(m_dta.get_dims()), dimsb(m_dtb.get_dims());
    dimsa.permute(m_tra.get_perm());
    dimsb.permute(m_trb.get_perm());
    return dimsa.equals(dimsb);
}


} // namespace libtensor

