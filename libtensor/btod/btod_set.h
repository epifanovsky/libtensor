#ifndef LIBTENSOR_BTOD_SET_H
#define LIBTENSOR_BTOD_SET_H

#include "../defs.h"
#include "../core/block_tensor_i.h"
#include "../core/block_tensor_ctrl.h"
#include "../core/orbit.h"
#include "../core/orbit_list.h"
#include "../tod/tod_set.h"
#include "bad_block_index_space.h"

namespace libtensor {


/**	\brief Sets all elements of a block tensor to a value preserving
        the symmetry
    \tparam N Tensor order.

    \ingroup libtensor_btod
 **/
template<size_t N>
class btod_set {
public:
    static const char *k_clazz; //!< Class name

private:
    double m_v; //!< Value

public:
    /** \brief Initializes the operation
        \param v Value to be assigned to the tensor elements.
     **/
    btod_set(double v = 0.0);

    /** \brief Performs the operation
        \param bt Output block tensor.
     **/
    void perform(block_tensor_i<N, double> &bt);

private:
    btod_set(const btod_set<N> &);
    const btod_set<N> &operator=(const btod_set<N> &);

};


template<size_t N>
const char *btod_set<N>::k_clazz = "btod_set<N>";


template<size_t N>
btod_set<N>::btod_set(double v) :

    m_v(v) {

}

template<size_t N>
void btod_set<N>::perform(block_tensor_i<N, double> &bt) {

    cpu_pool cpus(1);

    block_tensor_ctrl<N, double> ctrl(bt);

    orbit_list<N, double> ol(ctrl.req_symmetry());

    for(typename orbit_list<N, double>::iterator io = ol.begin();
        io != ol.end(); ++io) {

        index<N> bi(ol.get_index(io));
        if(m_v == 0.0) {
            ctrl.req_zero_block(bi);
        } else {
            dense_tensor_i<N, double> &blk = ctrl.req_block(bi);
            tod_set<N>(m_v).perform(cpus, blk);
            ctrl.ret_block(bi);
        }
    }
}


} // namespace libtensor

#endif // LIBTENSOR_BTOD_SET_H
