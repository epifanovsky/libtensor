#ifndef LIBTENSOR_BTOD_SCALE_H
#define LIBTENSOR_BTOD_SCALE_H

#include "../defs.h"
#include "../timings.h"
#include "../core/block_tensor_i.h"
#include "../core/block_tensor_ctrl.h"
#include "../core/orbit.h"
#include "../core/orbit_list.h"
#include <libtensor/dense_tensor/tod_scale.h>

namespace libtensor {


/** \brief Scales a block %tensor by a coefficient
    \tparam N Tensor order.

    \ingroup libtensor_btod
 **/
template<size_t N>
class btod_scale : public timings< btod_scale<N> > {
public:
    static const char *k_clazz; //!< Class name

private:
    block_tensor_i<N, double> &m_bt; //!< Block %tensor
    double m_c; //!< Scaling coefficient

public:
    /** \brief Initializes the operation
        \param bt Block %tensor.
        \param c Scaling coefficient.
     **/
    btod_scale(block_tensor_i<N, double> &bt, double c) :
        m_bt(bt), m_c(c) { }

    /** \brief Performs the operation
     **/
    void perform();
};


template<size_t N>
const char *btod_scale<N>::k_clazz = "btod_scale<N>";


template<size_t N>
void btod_scale<N>::perform() {

	btod_scale<N>::start_timer();

	try {

        block_tensor_ctrl<N, double> ctrl(m_bt);

        orbit_list<N, double> ol(ctrl.req_const_symmetry());
        for(typename orbit_list<N, double>::iterator io = ol.begin();
            io != ol.end(); io++) {

            index<N> idx(ol.get_index(io));
            if(ctrl.req_is_zero_block(idx)) continue;
            if(m_c == 0.0) {
                ctrl.req_zero_block(idx);
            } else {
                dense_tensor_i<N, double> &blk = ctrl.req_block(idx);
                tod_scale<N>(m_c).perform(blk);
                ctrl.ret_block(idx);
            }
        }

    } catch(...) {
        btod_scale<N>::stop_timer();
        throw;
    }

    btod_scale<N>::stop_timer();
}


} // namespace libtensor

#endif // LIBTENSOR_BTOD_SCALE_H
