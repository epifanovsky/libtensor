#ifndef LIBTENSOR_BTOD_DOTPROD_H
#define LIBTENSOR_BTOD_DOTPROD_H

#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/block_tensor/bto/bto_dotprod.h>
#include <libtensor/block_tensor/btod/btod_traits.h>
#include <libtensor/dense_tensor/tod_dotprod.h>

namespace libtensor {


struct btod_dotprod_traits : public bto_traits<double> {

    template<size_t N> struct to_dotprod_type {
        typedef tod_dotprod<N> type;
    };

};


template<size_t N>
class btod_dotprod : public bto_dotprod<N, btod_dotprod_traits> {
private:
    typedef bto_dotprod<N, btod_dotprod_traits> bto_dotprod_t;

public:
    /** \brief Initializes the first argument pair
            (identity permutation)
     **/
    btod_dotprod(block_tensor_i<N, double> &bt1,
        block_tensor_i<N, double> &bt2) : bto_dotprod_t(bt1, bt2) {

    }

    /** \brief Initializes the first argument pair
     **/
    btod_dotprod(block_tensor_i<N, double> &bt1,
        const permutation<N> &perm1, block_tensor_i<N, double> &bt2,
        const permutation<N> &perm2) : bto_dotprod_t(bt1, perm1, bt2, perm2) {

    }

    virtual ~btod_dotprod() { }

private:
    btod_dotprod(const btod_dotprod<N>&);
    const btod_dotprod<N> &operator=(const btod_dotprod<N>&);

};


} // namespace libtensor


#endif // LIBTENSOR_BTOD_DOTPROD_H
