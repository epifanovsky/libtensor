#ifndef LIBTENSOR_CTF_SYMMETRY_H
#define LIBTENSOR_CTF_SYMMETRY_H

#include <utility>
#include <vector>
#include <libtensor/core/permutation.h>

namespace libtensor {


/** \brief Carries permutational symmetry information for a CTF block
    \tparam N Tensor order.
    \tparam T Tensor element type.

    This data structure stores information about the permutational symmetry
    of a ctf_dense_tensor using symmetry components. The tensor is thus a sum
    of its components.

    Within each component, the symmetry is a direct product of permutational
    subgroups. Each permutational subgroup must be fully symmetric or
    antisymmetric with only pairwise index permutations supported. Furthermore,
    permutational symmetry in CTF (and therefore within each symmetry component
    here) is limited to adjacent tensor indices.

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
    typedef std::pair< sequence<N, unsigned>, sequence<N, unsigned> >
        symmetry_component_type;

private:
    std::vector<symmetry_component_type> m_sym; //!< Symmetry components

public:
    /** \brief Default constructor
     **/
    ctf_symmetry();

    /** \brief Initializing constructor
        \param grp Symmetry groups
        \param sym Symmetric (0) or antisymmetric (1) indicators
        \param jilk Specific case of (ijkl->jilk) symmetry
     **/
    ctf_symmetry(
        const sequence<N, unsigned> &grp,
        const sequence<N, unsigned> &sym,
        bool jilk = false);

    /** \brief Adds a symmetry component
        \param grp Symmetry groups
        \param sym Symmetric (0) or antisymmetric (1) indicators
     **/
    void add_component(
        const sequence<N, unsigned> &grp,
        const sequence<N, unsigned> &sym);

    /** \brief Returns the number of symmetry components
     **/
    size_t get_ncomp() const {
        return m_sym.size();
    }

    /** \brief Returns the symmetry subgroup index array
        \param icomp Component number
     **/
    const sequence<N, unsigned> &get_grp(size_t icomp) const {
        return m_sym[icomp].first;
    }

    /** \brief Returns the symmetric/antisymmetric indicator array
        \param icomp Component index
     **/
    const sequence<N, unsigned> &get_sym(size_t icomp) const {
        return m_sym[icomp].second;
    }

    /** \brief Returns true if given symmetry is a subgroup of this symmetry
     **/
    bool is_subgroup(const ctf_symmetry &other) const;

    /** \brief Applies permutation to the indices
     **/
    void permute(const permutation<N> &perm);

    /** \brief Exports symmetry in the CTF format
     **/
    void write(size_t icomp, int (&sym)[N]) const;

public:
    /** \brief Produces a compensation factor for A being symmetrized into
            higher symmetry of B by CTF
     **/
    static T symconv_factor(const ctf_symmetry<N, T> &syma, size_t icompa,
        const ctf_symmetry<N, T> &symb, size_t icompb);

    /** \brief Produces a compensation factor for A being permuted into
            higher symmetry by CTF
     **/
    static T symconv_factor(const ctf_symmetry<N, T> &sym, size_t icomp,
        const permutation<N> &perm);

    /** \brief Sets a global flag to disable/enable use of symmetry in CTF
     **/
    static void use_ctf_symmetry(bool use);

};


} // namespace libtensor

#endif // LIBTENSOR_CTF_SYMMETRY_H
