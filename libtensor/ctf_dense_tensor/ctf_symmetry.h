#ifndef LIBTENSOR_CTF_SYMMETRY_H
#define LIBTENSOR_CTF_SYMMETRY_H

#include <libtensor/core/permutation.h>

namespace libtensor {


/** \brief Carries permutational symmetry information for a CTF block
    \tparam N Tensor order.
    \tparam T Tensor element type.

    This data structure stores information about the permutational symmetry
    of a CTF tensor using the direct product of permutational subgroups.
    Each permutational subgroup must be fully symmetric or antisymmetric.
    Only pairwise index permutations are supported. Furthermore, permutational
    symmetry in CTF is limited to adjacent tensor indices.

    Permutational symmetry is stored as a combination of two arrays, index
    grouping and the symmetric/antisymmetric indicator for each subgroup.
    Each tensor index belongs to a symmetric group (which is comprised of
    just one index if it is nonsymmetric). The groups are numbered from 0
    to the total number of groups less one, which is equal to (N-1) when
    there is no symmetry at all. Group numbers are mapped into
    a 0 (symmetric) or 1 (antisymmetric) in the indicator array.

    The default constructor creates a group with no symmetry between indices.

    \ingroup libtensor_ctf_dense_tensor
 **/
template<size_t N, typename T>
class ctf_symmetry {
private:
    sequence<N, unsigned> m_grp; //!< Symmetry groups
    sequence<N, unsigned> m_sym; //!< Symmetric (0) or antisymmetric (1) groups

public:
    /** \brief Default constructor
     **/
    ctf_symmetry();

    /** \brief Initializing constructor
        \param grp Symmetry groups
        \param sym Symmetric (0) or antisymmetric (1) indicators
     **/
    ctf_symmetry(
        const sequence<N, unsigned> &grp,
        const sequence<N, unsigned> &sym);

    /** \brief Returns the symmetry subgroup index array
     **/
    const sequence<N, unsigned> &get_grp() const {
        return m_grp;
    }

    /** \brief Returns the symmetric/antisymmetric indicator array
     **/
    const sequence<N, unsigned> &get_sym() const {
        return m_sym;
    }

    /** \brief Returns true if given symmetry is a subgroup of this symmetry
     **/
    bool is_subgroup(const ctf_symmetry &other) const;

    /** \brief Applies permutation to the indices
     **/
    void permute(const permutation<N> &perm);

    /** \brief Exports symmetry in the CTF format
     **/
    void write(int (&sym)[N]) const;

};


} // namespace libtensor

#endif // LIBTENSOR_CTF_SYMMETRY_H
