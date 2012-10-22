#ifndef LIBTENSOR_GEN_BTO_DOTPROD_H
#define LIBTENSOR_GEN_BTO_DOTPROD_H

#include <list>
#include <vector>
#include <libtensor/core/noncopyable.h>
#include <libtensor/core/tensor_transf.h>
#include <libtensor/timings.h>
#include "gen_block_tensor_i.h"

namespace libtensor {


/** \brief Computes the dot product of two block tensors
    \tparam N Tensor order.
    \tparam Traits Block tensor operation traits.
    \tparam Timed Timed implementation.

    The dot product of two tensors is defined as the sum of elements of
    the element-wise product:

    \f[ c = \sum_i a_i b_i \f]

    This operation computes the dot product for a series of arguments.

    The traits class has to provide definitions for
    - \c element_type -- Type of data elements
    - \c bti_traits -- Type of block tensor interface traits class
    - \c template temp_block_type<N>::type -- Type of temporary tensor block
    - \c template to_dotprod_type<N>::type -- Type of tensor operation to_dotprod

    \ingroup libtensor_gen_bto
 **/
template<size_t N, typename Traits, typename Timed>
class gen_bto_dotprod : public timings<Timed>, public noncopyable {
public:
    //! Type of tensor elements
    typedef typename Traits::element_type element_type;

    //! Block tensor interface traits
    typedef typename Traits::bti_traits bti_traits;

    //! Type of read-only block
    typedef typename bti_traits::template rd_block_type<N>::type rd_block_type;

    //! Type of tensor transformation of result
    typedef tensor_transf<N, element_type> tensor_transf_type;

public:
    static const char *k_clazz; //!< Class name

private:
    struct arg {
        gen_block_tensor_rd_i<N, bti_traits> &bt1;
        gen_block_tensor_rd_i<N, bti_traits> &bt2;
        tensor_transf_type tr1;
        tensor_transf_type tr2;

        arg(gen_block_tensor_rd_i<N, bti_traits> &bt1_,
            const tensor_transf_type &tr1_,
            gen_block_tensor_rd_i<N, bti_traits> &bt2_,
            const tensor_transf_type &tr2_) :
            bt1(bt1_), bt2(bt2_), tr1(tr1_), tr2(tr2_) {

        }
    };

private:
    block_index_space<N> m_bis; //!< Block %index space of arguments
    std::list<arg> m_args; //!< Arguments

public:
    /** \brief Initializes the first argument pair
     **/
    gen_bto_dotprod(
            gen_block_tensor_rd_i<N, bti_traits> &bt1,
            const tensor_transf_type &tr1,
            gen_block_tensor_rd_i<N, bti_traits> &bt2,
            const tensor_transf_type &tr2);

    /** \brief Adds a pair of arguments
     **/
    void add_arg(
            gen_block_tensor_rd_i<N, bti_traits> &bt1,
            const tensor_transf_type &tr1,
            gen_block_tensor_rd_i<N, bti_traits> &bt2,
            const tensor_transf_type &tr2);

    /** \brief Computes the dot product for all argument pairs
     **/
    void calculate(std::vector<element_type> &v);
};


} // namespace libtensor


#endif // LIBTENSOR_GEN_BTO_DOTPROD_H
