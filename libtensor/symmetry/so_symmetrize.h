#ifndef LIBTENSOR_SO_SYMMETRIZE_H
#define LIBTENSOR_SO_SYMMETRIZE_H

#include "../core/symmetry.h"
#include "symmetry_operation_base.h"
#include "symmetry_operation_params.h"

namespace libtensor {

template<size_t N, typename T>
class so_symmetrize;

template<size_t N, typename T>
class symmetry_operation_params< so_symmetrize<N, T> >;

/**	\brief Adds a permutation %symmetry element to a group
	\tparam N Symmetry cardinality (%tensor order).
	\tparam T Tensor element type.

	\ingroup libtensor_symmetry
 **/
template<size_t N, typename T>
class so_symmetrize : public symmetry_operation_base< so_symmetrize<N, T> > {
private:
    typedef so_symmetrize<N, T> operation_t;
    typedef symmetry_operation_dispatcher<operation_t> dispatcher_t;

private:
    const symmetry<N, T> &m_sym1; //!< Symmetry group
    mask<N> m_msk; //!< Mask of indexes to be symmetrized
    bool m_symm; //!< Symmetric/anti-symmetric flag (for pair permutations)

public:
    /**	\brief Initializes the operation
		\param sym1 Symmetry container.
		\param msk Indexes to be symmetrized
		\param symm Symmetric (true)/anti-symmetric (false)
     **/
    so_symmetrize(const symmetry<N, T> &sym1, const mask<N> &msk, bool symm) :
        m_sym1(sym1), m_msk(msk), m_symm(symm) { }

    /**	\brief Performs the operation
		\param sym2 Destination %symmetry container.
     **/
    void perform(symmetry<N, T> &sym2);

private:
    so_symmetrize(const so_symmetrize<N, T>&);
    const so_symmetrize<N, T> &operator=(const so_symmetrize<N, T>&);
};

template<size_t N, typename T>
class symmetry_operation_params< so_symmetrize<N, T> > :
public symmetry_operation_params_i {

public:
    const symmetry_element_set<N, T> &grp1; //!< Symmetry group 1
    mask<N> msk; //!< Mask of indexes
    bool symm; //!< Symmetrize/anti-symmetrize flag (for pair permutation)
    symmetry_element_set<N, T> &grp2; //!< Symmetry group 2 (output)

public:
    symmetry_operation_params(
            const symmetry_element_set<N, T> &grp1_, const mask<N> &msk_,
            bool symm_, symmetry_element_set<N, T> &grp2_) :

                grp1(grp1_), msk(msk_), symm(symm_), grp2(grp2_) { }

    virtual ~symmetry_operation_params() { }
};

} // namespace libtensor

#include "so_symmetrize_handlers.h"

#endif // LIBTENSOR_SO_SYMMETRIZE_H
