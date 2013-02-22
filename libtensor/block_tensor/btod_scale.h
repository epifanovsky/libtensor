#ifndef LIBTENSOR_BTOD_SCALE_H
#define LIBTENSOR_BTOD_SCALE_H

#include <libtensor/block_tensor/btod_traits.h>
#include <libtensor/gen_block_tensor/gen_bto_scale.h>

namespace libtensor {


/** \brief Scales a block %tensor by a coefficient
    \tparam N Tensor order.

    \ingroup libtensor_block_tensor_btod
 **/
template<size_t N>
class btod_scale : public noncopyable {
public:
    static const char *k_clazz; //!< Class name

private:
    gen_bto_scale< N, btod_traits, btod_scale<N> > m_gbto;

public:
    /** \brief Initializes the operation
        \param bt Block %tensor.
        \param c Scaling coefficient.
     **/
    btod_scale(block_tensor_i<N, double> &bt, const scalar_transf<double> &c) :
        m_gbto(bt, c) { }

    /** \brief Initializes the operation
        \param bt Block %tensor.
        \param c Scaling coefficient.
     **/
    btod_scale(block_tensor_i<N, double> &bt, double c) :
        m_gbto(bt, scalar_transf<double>(c)) { }

    /** \brief Performs the operation
     **/
    void perform() {
        m_gbto.perform();
    }
};


template<size_t N>
const char *btod_scale<N>::k_clazz = "btod_scale<N>";

} // namespace libtensor

#endif // LIBTENSOR_BTOD_SCALE_H
