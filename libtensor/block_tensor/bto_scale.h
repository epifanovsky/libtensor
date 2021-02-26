#ifndef LIBTENSOR_BTO_SCALE_H
#define LIBTENSOR_BTO_SCALE_H

#include <libtensor/block_tensor/bto_traits.h>
#include <libtensor/gen_block_tensor/gen_bto_scale.h>

namespace libtensor {


/** \brief Scales a block tensor by a coefficient
    \tparam N Tensor order.

    \ingroup libtensor_block_tensor_btod
 **/
template<size_t N, typename T>
class bto_scale : public noncopyable {
public:
    static const char k_clazz[]; //!< Class name

private:
    gen_bto_scale< N, bto_traits<T>, bto_scale<N, T> > m_gbto;

public:
    /** \brief Initializes the operation
        \param bt Block tensor.
        \param c Scaling coefficient.
     **/
    bto_scale(block_tensor_i<N, T> &bt, const scalar_transf<T> &c) :
        m_gbto(bt, c) { }

    /** \brief Initializes the operation
        \param bt Block tensor.
        \param c Scaling coefficient.
     **/
    bto_scale(block_tensor_i<N, T> &bt, T c) :
        m_gbto(bt, scalar_transf<T>(c)) { }

    /** \brief Performs the operation
     **/
    void perform() {
        m_gbto.perform();
    }

};


} // namespace libtensor

#endif // LIBTENSOR_BTO_SCALE_H
