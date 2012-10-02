#ifndef LIBTENSOR_GEN_BTO_H
#define LIBTENSOR_GEN_BTO_H

/** \page gen_bto Generalized block tensors operations

    The generalized block tensors and block tensor operations provide
    building blocks for specialized block tensors and operations which employ
    specific types of tensors and data types. The tensor types and element
    types have to be provided via traits classes.

    The main traits class is the block tensor interface traits class. Any
    implementation of block tensors has to define this class with the
    following member types:
    - element_type -- Type of the data elements
    - template<N> rd_block_type::type -- Type of read-only tensor block
        interface
    - template<N> wr_block_type::type -- Type of read-write tensor block
        interface

    Additionally, there is the block tensor traits class which comprises the
    member types
    - element_type -- Type of data elements
    - allocator_type -- Type of allocator to use for allocating blocks
    - bti_traits -- Type of block tensor interface traits class
    - template<N> block_type::type -- Type of read-write tensor blocks
    - template<N> block_factory_type::type -- Type of block factory to create
        blocks

    The generalized block tensor operations employ another traits class
    with member types
    - element_type -- Type of data elements
    - bti_traits -- Type of block tensor interface traits class
    - template<N> temp_block_tensor_type::type -- Type of a temporary block
        tensor
    - template<N> to_add_type::type -- Type of tensor operation for addition
    - template<N, Functor> to_apply_type::type -- Type of tensor operation
        for element-wise application of a function
    - template<N> to_compare_type::type -- Type of tensor comparison
    - template<N, M, K> to_contract2_type::type -- Type of tensor operation
        for contraction of two tensors
    - template<N> to_copy_type::type -- Type of tensor operation for copy
    - template<N, M> to_diag_type::type -- Type of tensor operation for
        taking a generalized diagonal
    - template<N, M> to_dirsum_type::type -- Type of tensor operation for
        direct sum
    - template<N> to_dotprod_type::type -- Type of tensor operation for
        dot product
    - template<N, M, K> to_ewmult2_type::type -- Type of tensor operation for
        generalized element-wise multiplication
    - template<N, M> to_extract_type::type -- Type of tensor operation for
        extraction of lower-rank tensors
    - template<N> to_mult_type::type -- Type of tensor operation for
        element-wise multiplication of two tensors
    - template<N> to_mult1_type::type -- Type of tensor operation for
        element-wise multiplication of one tensor to another tensor
    - template<N> to_random_type::type -- Type of tensor operation for filling
        a tensor with random data
    - template<N> to_scale_type::type -- Type of tensor operation for scalar
        transformation of a tensor
    - template<N, M> to_scatter_type::type -- Type of tensor operation for
        distributing elements of a tensor into a higher-rank tensor
    - template<N> to_select_type::type -- Type of tensor operation for
        selecting tensor elements according to a compare policy
    - template<N> to_set_diag_type::type -- Type of tensor operation for
        setting all diagonal tensor elements to a certain value
    - template<N> to_set_elem_type::type -- Type of tensor operation for
        setting a specific tensor element to a certain value
    - template<N> to_set_type::type -- Type of tensor operation for setting
        all tensor elements to a certain value
    - template<N> to_trace_type::type -- Type of tensor operation for computing
        the trace of a tensor
    - template<N> to_vmpriority_type::type -- Type of tensor operation for
        setting and unsetting of memory priority
    This traits class also has to have the following member functions defined
    - bool is_zero(element_type) -- Return if the element is zero
    - element_type zero() -- Return the zero element
    - element_type identity() -- Return the identity element

    \ingroup libtensor_gen_bto
 **/


/** \page gen_toi Generalized tensors operation interfaces

    The generalized block tensor operations expect the following interfaces
    for the tensor operations that do the actual work (Remark: class names are
    defined in the traits classes and can be chosen freely).

    - Class interface for tensor addition
    \code
    template<size_t N>
    class to_add {
    public:
        to_add(rd_block_type<N> &ta,
            const tensor_transf<N, element_type> &tr =
                tensor_transf<N, element_type());

        void add_op(rd_block_type<N> &ta,
            const tensor_transf<N, element_type> &tr =
                tensor_transf<N, element_type())

        void perform(bool zero, wr_block_type<N> &tb);
    };
    \endcode
    - Class interface for applying a function to a tensor (element-wise)
    \code
    template<size_t N, typename Functor>
    class to_apply {
    public:
        to_apply(rd_block_type<N> &ta, const Functor &fn,
            const scalar_transf<element_type> &tr1 =
                scalar_transf<element_type>(),
            const tensor_transf<N, element_type> &tr2 =
                tensor_transf<N, element_type());

        void perform(bool zero, wr_block_type<N> &tb);
    };
    \endcode
    - Class interface for comparing two tensors
    \code
    template<size_t N>
    class to_compare {
    public:
        to_compare(rd_block_type<N> &ta, rd_block_type<N> &tb,
            const element_type &thresh);

        bool compare();

        const index<N> &get_diff_index() const;

        const element_type &get_diff_elem_1() const;

        const element_type &get_diff_elem_2() const;
    };
    \endcode
    - Class interface for contraction of two tensors
    \code
    template<size_t N, size_t M, size_t K>
    class to_contract2 {
    public:
        to_contract2(const contraction2<N, M, K> &contr,
            rd_block_type<N + K> &ta, rd_block_type<M + K> &tb,
            const scalar_transf<element_type> &tr =
                scalar_transf<element_type>());

        void add_args(const contraction2<N, M, K> &contr,
            rd_block_type<N + K> &ta, rd_block_type<M + K> &tb,
            const scalar_transf<element_type> &tr);

        void perform(bool zero, wr_block_type<N + M> &tc);
    };
    \endcode
    - Class interface for tensor copy
    \code
    template<size_t N>
    class to_copy {
    public:
        to_copy(rd_block_type<N> &ta,
            const tensor_transf<N, element_type> &tr =
                tensor_transf<N, element_type());

        void perform(bool zero, wr_block_type<N> &tb);
    };
    \endcode
    - Class interface for extracting the diagonal of a tensor
    \code
    template<size_t N, size_t M>
    class to_diag {
    public:
        to_diag(rd_block_type<N> &ta, const mask<N> &m,
            const tensor_transf<N - M + 1, element_type> &tr =
                tensor_transf<N - M + 1, element_type());

        void perform(bool zero, wr_block_type<N - M + 1> &tb);
    };
    \endcode
    - Class interface for forming the direct sum of two tensors
    \code
    template<size_t N, size_t M>
    class to_dirsum {
    public:
        to_dirsum(rd_block_type<N> &ta,
            const scalar_transf<element_type> &ka,
            rd_block_type<N> &tb,
            const scalar_transf<element_type> &kb,
            const tensor_transf<N + M, element_type> &trc =
                tensor_transf<N + M, element_type());

        void perform(bool zero, wr_block_type<N> &tc);
    };
    \endcode
    - Class interface for computing the inner (dot) product
    \code
    template<size_t N, size_t M>
    class to_dotprod {
    public:
        to_dotprod(rd_block_type<N> &ta,
            const tensor_transf<N, element_type> &tra,
            rd_block_type<N> &tb,
            const tensor_transf<N, element_type> &trb);

        element_type calculate();
    };
    \endcode
    - Class interface for general element-wise multiplication of two tensors
    \code
    template<size_t N, size_t M, size_t K>
    class to_ewmult2 {
    public:
        to_ewmult2(rd_block_type<N + K> &ta,
            const tensor_transf<N + K, element_type> &tra,
            rd_block_type<M + K> &tb,
            const tensor_transf<M + K, element_type> &trb,
            const tensor_transf<N + M + K, element_type> &trc =
                tensor_transf<N + M + K, element_type());

        void perform(bool zero, wr_block_type<N + M + K> &tc);
    };
    \endcode
    - Class interface for extracting a lower-rank tensor
    \code
    template<size_t N, size_t M>
    class to_extract {
    public:
        to_extract(rd_block_type<N> &ta,
            const mask<N> &m, const index<N> &idx,
            const tensor_transf<N - M, element_type> &trb =
                tensor_transf<N - M, element_type());

        void perform(bool zero, wr_block_type<N - M> &tb);
    };
    \endcode
    - Class interface for element-wise multiplication or division of two tensors
    \code
    template<size_t N>
    class to_mult {
    public:
        to_mult(rd_block_type<N> &ta,
            const tensor_transf<N, element_type> &tra,
            rd_block_type<N> &tb,
            const tensor_transf<N, element_type> &trb,
            bool recip,
            const scalar_transf<element_type> &trc =
                scalar_transf<element_type());

        void perform(bool zero, wr_block_type<N> &tc);
    };
    \endcode
    - Class interface for element-wise multiplication or division to a tensor
    \code
    template<size_t N>
    class to_mult1 {
    public:
        to_mult1(rd_block_type<N> &tb,
            const tensor_transf<N, element_type> &trb,
            bool recip,
            const scalar_transf<element_type> &trc =
                scalar_transf<element_type());

        void perform(bool zero, wr_block_type<N> &tc);
    };
    \endcode
    - Class interface for filling a tensor with random data
    \code
    template<size_t N>
    class to_random {
    public:
        to_random(const scalar_transf<element_type> &c =
                scalar_transf<element_type());

        void perform(bool zero, wr_block_type<N> &tc);
    };
    \endcode
    - Class interface for applying a scalar transformation to a tensor
    \code
    template<size_t N>
    class to_scale {
    public:
        to_scale(const scalar_transf<element_type> &c);

        void perform(wr_block_type<N> &tc);
    };
    \endcode
    - Class interface for distributing a tensor into a higher-rank tensor
    \code
    template<size_t N>
    class to_scatter {
    public:
        to_scatter(rd_block_type<N> &ta,
            const tensor_transf<N, element_type> &trc);

        void perform(bool zero, wr_block_type<N> &tc);
    };
    \endcode
    - Class interface for selecting tensor elements accoring to a compare
      policy
    \code
    template<size_t N, typename ComparePolicy>
    class to_select {
    public:
        to_select(rd_block_type<N> &ta,
            tensor_transf<N, element_type> &tra,
            ComparePolicy cmp);

        void perform(std::list< tensor_element<element_type> > &, size_t n);
    };
    \endcode
    - Class interface for setting the diagonal of a tensor
    \code
    template<size_t N>
    class to_set_diag {
    public:
        to_set_diag(const element_type& v);

        void perform(wr_block_type<N> &tc);
    };
    \endcode
    - Class interface for setting a single element of a tensor
    \code
    template<size_t N>
    class to_set_elem {
    public:
        void perform(wr_block_type<N> &tc, const index<N> &idx,
            const element_type &v);
    };
    \endcode
    - Class interface for setting all elements of a tensor
    \code
    template<size_t N>
    class to_set {
    public:
        to_set(const element_type &c);

        void perform(wr_block_type<N> &tc);
    };
    \endcode
    - Class interface for computing the trace of a tensor
    \code
    template<size_t N>
    class to_trace {
    public:
        to_trace(rd_block_type<N> &t);

        element_type calculate();
    };
    \endcode
    - Class interface for setting the virtual memory priority of the tensor
    \code
    template<size_t N>
    class to_vmpriority {
    public:
        to_vmpriority(rd_block_type<N> &t);

        void set_priority();

        void unset_priority();
    };
    \endcode


    \ingroup libtensor_gen_bto
 **/

#include "gen_block_stream_i.h"
#include "gen_block_tensor_ctrl.h"
#include "gen_block_tensor_i.h"
#include "gen_block_tensor.h"


#include "gen_bto_add.h"
#include "gen_bto_apply.h"
#include "gen_bto_contract2.h"
#include "gen_bto_copy.h"
#include "gen_bto_diag.h"
#include "gen_bto_dirsum.h"
#include "gen_bto_mult.h"
#include "gen_bto_set.h"


#endif // LIBTENSOR_GEN_BTO_H

