#ifndef LIBTENSOR_BTOD_SET_DIAG_H
#define LIBTENSOR_BTOD_SET_DIAG_H

#include "../defs.h"
#include "../exception.h"
#include "../core/block_tensor_i.h"
#include "../core/block_tensor_ctrl.h"
#include "../core/orbit.h"
#include <libtensor/dense_tensor/tod_set.h>
#include <libtensor/dense_tensor/tod_set_diag.h>

namespace libtensor {


/** \brief Assigns the diagonal elements of a block %tensor to a value
    \tparam N Tensor order.

    This operation sets the diagonal elements of a block %tensor to a value
    without affecting all the off-diagonal elements.

    \ingroup libtensor_btod
 **/
template<size_t N>
class btod_set_diag {
public:
    static const char *k_clazz; //!< Class name

private:
    double m_v; //!< Value

public:
    //!    \name Construction and destruction
    //@{

    /** \brief Initializes the operation
        \param v Tensor element value (default 0.0).
     **/
    btod_set_diag(double v = 0.0);

    //@}

    //!    \name Operation
    //@{

    /** \brief Performs the operation
     **/
    void perform(block_tensor_i<N, double> &bt);

    //@}
};


template<size_t N>
const char *btod_set_diag<N>::k_clazz = "btod_set_diag<N>";


template<size_t N>
btod_set_diag<N>::btod_set_diag(double v) : m_v(v) {

}


template<size_t N>
void btod_set_diag<N>::perform(block_tensor_i<N, double> &bt) {

    static const char *method = "perform(block_tensor_i<N, double>&)";

    const block_index_space<N> &bis = bt.get_bis();
    size_t t = bis.get_type(0);
    for(size_t i = 1; i < N; i++) {
        if(bis.get_type(i) != t) {
            throw bad_parameter(g_ns, k_clazz, method, __FILE__,
                __LINE__, "Invalid block tensor dimension.");
        }
    }

    block_tensor_ctrl<N, double> ctrl(bt);

    dimensions<N> dims(bis.get_block_index_dims());
    size_t n = dims[0];
    index<N> idx;
    for(size_t i = 0; i < n; i++) {

        for(size_t j = 0; j < N; j++) idx[j] = i;

        abs_index<N> aidx(idx, dims);
        orbit<N, double> o(ctrl.req_const_symmetry(), idx);
        if(!o.is_allowed()) continue;
        if(o.get_abs_canonical_index() != aidx.get_abs_index())
            continue;

        if(ctrl.req_is_zero_block(idx)) {
            if(m_v != 0.0) {
                dense_tensor_i<N, double> &blk = ctrl.req_block(idx);
                tod_set<N>(0.0).perform(blk);
                tod_set_diag<N>(m_v).perform(blk);
                ctrl.ret_block(idx);
            }
        } else {
            dense_tensor_i<N, double> &blk = ctrl.req_block(idx);
            tod_set_diag<N>(m_v).perform(blk);
            ctrl.ret_block(idx);
        }
    }
}


} // namespace libtensor

#endif // LIBTENSOR_BTOD_SET_DIAG_H
