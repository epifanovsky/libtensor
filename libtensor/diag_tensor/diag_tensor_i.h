#ifndef LIBTENSOR_DIAG_TENSOR_I_H
#define LIBTENSOR_DIAG_TENSOR_I_H

#include "diag_tensor_space.h"

namespace libtensor {


template<size_t N, typename T> class diag_tensor_base_ctrl;
template<size_t N, typename T> class diag_tensor_rd_ctrl;
template<size_t N, typename T> class diag_tensor_wr_ctrl;


/** \brief Generalized diagonal tensor interface (abstract base class)
    \tparam N Tensor order.
    \tparam T Tensor element type.

    <b>General information</b>

    The diagonal tensor is a kind of sparse tensors that have a specific
    sparsity pattern. The entries that are allowed to take non-zero values
    are restricted by the sparse tensor space (diag_tensor_space). In the case
    of the generalized diagonal tensors, one may specify those dimensions,
    along which only diagonal entries exist. If no restrictions are specified,
    the diagonal tensor becomes the usual dense tensor (dense_tensor_base_i).

    For example, in the case of a square matrix \f$ a_{ij} \f$ with no
    restrictions all entries \f$ a_{ij} \f$ can be non-zero. However, if the
    two dimensions of the matrix are connected, only the diagonal elements
    \f$ a_{ii} \f$ are allowed.

    Tensors are more general than matrices, and so are the restrictions. One
    or more separate diagonal can be specified, and those diagonals can be
    multidimensional (more than two) diagonals. Some indexes are allowed to be
    free (unrestricted) so they can take any values.

    <b>Data format</b>

    The diagonal tensor is a combination of the components that arise from
    different subspaces of diag_tensor_space. Those components are stored
    in independent arrays.

    Each component is first "flattened", so that the indexes of each restricted
    diagonal combine to become just one index. The surviving index takes
    the position of the first index of that diagonal. The resulting flattened
    tensor has the same or smaller order and is dense. (It has the same order
    if there are no restricted diagonals, that is the tensor is dense from
    the beginning.) The reduced-order dense tensor is stored in the regular
    vectorized dense tensor format (see dense_tensor_base_i).

    For example, tensor \f$ a_{iijj} \f$ will be flattened to \f$ a_{ij} \f$,
    tensor \f$ a_{ijij} \f$ also becomes \f$ a_{ij} \f$, and tensor
    \f$ a_{ikjk} \f$ is stored as \f$ a_{ikj} \f$.

    \sa dense_tensor_base_i, diag_tensor_space, diag_tensor_rd_i,
        diag_tensor_wr_i, diag_tensor_base_ctrl

    \ingroup libtensor_diag_tensor
 **/
template<size_t N, typename T>
class diag_tensor_base_i {
    friend class diag_tensor_base_ctrl<N, T>;

public:
    /** \brief Virtual destructor
     **/
    virtual ~diag_tensor_base_i() { }

    /** \brief Returns the space of the tensor
     **/
    virtual const diag_tensor_space<N> &get_space() const = 0;

};


template<size_t N, typename T>
class diag_tensor_rd_i : virtual public diag_tensor_base_i<N, T> {
    friend class diag_tensor_rd_ctrl<N, T>;
};


template<size_t N, typename T>
class diag_tensor_wr_i : virtual public diag_tensor_base_i<N, T> {
    friend class diag_tensor_wr_ctrl<N, T>;
};


template<size_t N, typename T>
class diag_tensor_i :
    public diag_tensor_rd_i<N, T>, public diag_tensor_wr_i<N, T> {

};


} // namespace libtensor

#endif // LIBTENSOR_DIAG_TENSOR_I_H

