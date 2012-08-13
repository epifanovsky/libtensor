#ifndef LIBTENSOR_BTOD_TRAITS_H
#define LIBTENSOR_BTOD_TRAITS_H

#include <libtensor/block_tensor/bto/bto_traits.h>
#include <libtensor/core/block_tensor_i.h>
#include <libtensor/core/block_tensor_ctrl.h>
#include <libtensor/dense_tensor/dense_tensor_i.h>
#include <libtensor/dense_tensor/tod_copy.h>
#include <libtensor/dense_tensor/tod_set.h>

namespace libtensor {

template<>
struct bto_traits<double> {

    //! BTO traits type required by additive bto
    typedef bto_traits<double> additive_bto_traits;

    //! Element type
    typedef double element_type;

    //! Type of block tensor
    template<size_t N> struct block_tensor_type {
        typedef block_tensor_i<N, double> type;
    };

    //! Type of block tensor control
    template<size_t N> struct block_tensor_ctrl_type {
        typedef block_tensor_ctrl<N, double> type;
    };

    //! Type of block of block tensors
    template<size_t N> struct block_type {
        typedef dense_tensor_i<N, double> type;
    };

    template<size_t N> struct to_copy_type {
        typedef tod_copy<N> type;
    };

    template<size_t N> struct to_set_type {
        typedef tod_set<N> type;
    };

    static bool is_zero(double d) {
        return d == 0.0;
    }

    static double zero() {
        return 0.0;
    }

    static double identity() {
        return 1.0;
    }
};


} // namespace libtensor

#endif // LIBTENSOR_BTOD_TRAITS_H
