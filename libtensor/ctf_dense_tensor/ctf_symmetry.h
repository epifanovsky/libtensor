#ifndef LIBTENSOR_CTF_SYMMETRY_H
#define LIBTENSOR_CTF_SYMMETRY_H

#include <libtensor/core/permutation.h>
#include <libtensor/core/transf_list.h>

namespace libtensor {


/** \brief Carries permutational symmetry information for a CTF block
    \tparam N Tensor order.
    \tparam T Tensor element type.

    \ingroup libtensor_ctf_dense_tensor
 **/
template<size_t N, typename T>
class ctf_symmetry {
public:
    /** \brief Default constructor
     **/
    ctf_symmetry();

    /** \brief Builds the symmetry object from a list of transformations
     **/
    void build(const transf_list<N, T> &trl);

    /** \brief Applies permutation to the indices
     **/
    void permute(const permutation<N> &perm);

    /** \brief Exports symmetry in the CTF format
     **/
    void write(int (&sym)[N]) const;

};


} // namespace libtensor

#endif // LIBTENSOR_CTF_SYMMETRY_H
