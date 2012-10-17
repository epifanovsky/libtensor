#ifndef LIBTENSOR_GEN_BTO_SCALE_H
#define LIBTENSOR_GEN_BTO_SCALE_H

#include <libtensor/defs.h>
#include <libtensor/timings.h>
#include <libtensor/core/orbit.h>
#include <libtensor/core/orbit_list.h>
#include "gen_block_tensor_ctrl.h"
#include "gen_block_tensor_i.h"

namespace libtensor {


/** \brief Apply a scalar transformation to a block %tensor
    \tparam N Tensor order.
    \tparam Traits Block tensor operation traits.
    \tparam Timed Timed implementation.

    \ingroup libtensor_btod
 **/
template<size_t N, typename Traits, typename Timed>
class gen_bto_scale : public timings<Timed>, public noncopyable {
public:
    //! Type of tensor elements
    typedef typename Traits::element_type element_type;

    //! Block tensor interface traits
    typedef typename Traits::bti_traits bti_traits;

    //! Type of read-only block
    typedef typename
            bti_traits::template rd_block_type<N>::type rd_block_type;

    //! Type of write-only block
    typedef typename
            bti_traits::template wr_block_type<N>::type wr_block_type;

public:
    static const char *k_clazz; //!< Class name

private:
    gen_block_tensor_i<N, bti_traits> &m_bt; //!< Block %tensor
    scalar_transf<element_type> m_c; //!< Scaling coefficient

public:
    /** \brief Initializes the operation
        \param bt Block %tensor.
        \param c Scaling coefficient.
     **/
    gen_bto_scale(gen_block_tensor_i<N, bti_traits> &bt,
            const scalar_transf<element_type> &c) :
        m_bt(bt), m_c(c) { }

    /** \brief Performs the operation
     **/
    void perform();
};


template<size_t N, typename Traits, typename Timed>
const char *gen_bto_scale<N, Traits, Timed>::k_clazz =
        "gen_bto_scale<N, Traits, Timed>";


template<size_t N, typename Traits, typename Timed>
void gen_bto_scale<N, Traits, Timed>::perform() {

    typedef typename Traits::template to_scale_type<N>::type to_scale_type;

	gen_bto_scale::start_timer();

	try {

        gen_block_tensor_ctrl<N, bti_traits> ctrl(m_bt);

        orbit_list<N, element_type> ol(ctrl.req_const_symmetry());
        for(typename orbit_list<N, element_type>::iterator io = ol.begin();
                io != ol.end(); io++) {

            index<N> idx(ol.get_index(io));
            if(ctrl.req_is_zero_block(idx)) continue;

            if(m_c.is_zero()) {
                ctrl.req_zero_block(idx);
            } else {
                wr_block_type &blk = ctrl.req_block(idx);
                to_scale_type(m_c).perform(blk);
                ctrl.ret_block(idx);
            }
        }

    } catch(...) {
        gen_bto_scale::stop_timer();
        throw;
    }

    gen_bto_scale::stop_timer();
}


} // namespace libtensor

#endif // LIBTENSOR_GEN_BTO_SCALE_H
