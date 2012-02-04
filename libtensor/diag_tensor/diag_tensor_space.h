#ifndef LIBTENSOR_DIAG_TENSOR_SPACE_H
#define LIBTENSOR_DIAG_TENSOR_SPACE_H

#include <vector>
#include <libtensor/core/dimensions.h>
#include <libtensor/core/mask.h>

namespace libtensor {


/** \brief Specifies a subspace in a general diagonal tensor space

    The subspace is defined by one or more nonintersecting sets of tensor
    indexes. Along each set only diagonal entries are allowed.

    \sa diag_tensor_base_i, diag_tensor_space

    \ingroup libtensor_diag_tensor
 **/
template<size_t N>
class diag_tensor_subspace {
public:
    static const char *k_clazz; //!< Class name

private:
    std::vector< mask<N> > m_diag; //!< Diagonal masks
    mask<N> m_msk; //!< Total mask

public:
    /** \brief Initializes the subspace
        \param n Number of restricted diagonals.
     **/
    diag_tensor_subspace(size_t n);

    /** \brief Returns the number of diagonals in the subspace
     **/
    size_t get_ndiag() const {
        return m_diag.size();
    }

    /** \brief Returns the combined mask of all restricted diagonals
     **/
    const mask<N> &get_total_mask() const {
        return m_msk;
    }

    /** \brief Returns the mask of the given restricted diagonal
     **/
    const mask<N> &get_diag_mask(size_t n) const;

    /** \brief Sets the mask of the given restricted diagonal
     **/
    void set_diag_mask(size_t n, const mask<N> &msk);

};


/** \brief Specifies the space of a general diagonal tensor
    \tparam N Tensor order.

    Diagonal tensor space specifies overall dimensions of the tensor returned
    by get_dims() and each of the component subspaces.

    The number of subspaces is returned by get_nsubspaces(), and each subspace
    can be obtained by calling get_subspace().

    Any number of subspaces can be specified, and the tensor space is the union
    of all the spaces. If an entry is allowed by any of the subspaces, it is
    allowed by the space.

    Upon addition, each subspace is assigned an integer, which can later be
    used to identify that subspace within the space. The numbers are not
    guaranteed to be consecutive or to come from a certain range. However,
    each subspace's id is guaranteed to be the same as long as that subspace
    is a part of the space. Removed subspace's number can be reused.

    \sa diag_tensor_base_i, diag_tensor_subspace

    \ingroup libtensor_diag_tensor
 **/
template<size_t N>
class diag_tensor_space {
public:
    static const char *k_clazz; //!< Class name

private:
    dimensions<N> m_dims; //!< Tensor dimensions
    size_t m_nss; //!< Number of subspaces
    std::vector< diag_tensor_subspace<N>* > m_ss; //!< Subspaces

public:
    /** \brief Initializes diag tensor space
     **/
    diag_tensor_space(const dimensions<N> &dims);

    /** \brief Copy constructor
     **/
    diag_tensor_space(const diag_tensor_space &other);

    /** \brief Destructor
     **/
    ~diag_tensor_space();

    /** \brief Returns tensor dimensions
     **/
    const dimensions<N> &get_dims() const {
        return m_dims;
    }

    /** \brief Returns the number of subspaces in the space
     **/
    size_t get_nsubspaces() const {
        return m_nss;
    }

    /** \brief Returns all subspace numbers into a vector
     **/
    void get_all_subspaces(std::vector<size_t> &ssn) const;

    /** \brief Returns a subspace by its number
     **/
    const diag_tensor_subspace<N> &get_subspace(size_t n) const;

    /** \brief Returns the number of non-zero entries allowed by a subspace
     **/
    size_t get_subspace_size(size_t n) const;

    /** \brief Adds a subspace, returns its number
     **/
    size_t add_subspace(const diag_tensor_subspace<N> &ss);

    /** \brief Removes a subspace by its number
     **/
    void remove_subspace(size_t n);

};


} // namespace libtensor

#endif // LIBTENSOR_DIAG_TENSOR_SPACE_H

