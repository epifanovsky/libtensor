#ifndef LIBTENSOR_EXPR_NODE_PRODUCT_H
#define LIBTENSOR_EXPR_NODE_PRODUCT_H

#include <vector>
#include <libtensor/expr/dag/node.h>

namespace libtensor {
namespace expr {


/** \brief Expression node: product of tensors with or without contraction

    This expression node represents the product of tensors (or just one tensor
    as a special case) with or without summation. Tensor operations that fall
    into this category are contractions, elementwise products, slices,
    generalized trace, etc.

    It is assumed that the dimensionality of each tensor arguments is known
    from the adjacent nodes in the graph.

    The product is represented by a concatenated array of the indices of the
    first argument, followed by the second argument, and so on. A separate array
    of indices lists those to be summed over (contracted).

    Examples:

    Elementwise multiplication A(ij) B(ij): I = ( 0 1 0 1 ); C = ( )

    Contraction sum(k) A(ik) B(kj): I = ( 0 2 2 1 ); C = ( 2 )
    Contraction sum(cd) A(icjd) B(acbd):
        I = ( 0 4 1 5 2 4 3 5 ); C = ( 4 5 )

    Diagonal extraction A(ii): I = ( 0 0 ); C = ( )

    Trace sum(i) A(ii): I = ( 0 0 ); C = ( 0 )

    \ingroup libtensor_expr
 **/
class node_product : public node {
private:
    std::vector<size_t> m_idx; //!< Tensor indices
    std::vector<size_t> m_cidx; //!< Contracted (inner) indices

public:
    /** \brief Creates a tensor product node (without contraction)
        \param op Operation name.
        \param n Order of result.
        \param idx Tensor indices.
     **/
    node_product(
        const std::string &op,
        size_t n,
        const std::vector<size_t> &idx);

    /** \brief Creates a tensor product node (with contraction)
        \param op Operation name.
        \param n Order of result.
        \param idx Tensor indices.
        \param cidx Contracted (inner) indices.
     **/
    node_product(
        const std::string &op,
        size_t n,
        const std::vector<size_t> &idx,
        const std::vector<size_t> &cidx);

    /** \brief Virtual destructor
     **/
    virtual ~node_product() { }

    /** \brief Returns the array of tensor indices
     **/
    const std::vector<size_t> &get_idx() const {
        return m_idx;
    }

    /** \brief Returns the array of contracted indices
     **/
    const std::vector<size_t> &get_cidx() const {
        return m_cidx;
    }

    /** \brief Builds the sequence of indices in the output into an array
     **/
    void build_output_indices(std::vector<size_t> &oidx) const;

private:
    void check();

};


} // namespace expr
} // namespace libtensor

#endif // LIBTENSOR_EXPR_NODE_PRODUCT_H
