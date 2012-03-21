#ifndef LIBTENSOR_SO_SYMMETRIZE3_H
#define LIBTENSOR_SO_SYMMETRIZE3_H

#include "../core/symmetry.h"
#include "symmetry_operation_base.h"
#include "symmetry_operation_params.h"

namespace libtensor {


template<size_t N, typename T>
class so_symmetrize3;

template<size_t N, typename T>
class symmetry_operation_params< so_symmetrize3<N, T> >;


/**	\brief Adds three-index permutation %symmetry element to a group
	\tparam N Symmetry cardinality (%tensor order).
	\tparam T Tensor element type.

	\ingroup libtensor_symmetry
 **/
template<size_t N, typename T>
class so_symmetrize3 : public symmetry_operation_base< so_symmetrize3<N, T> > {
private:
    typedef so_symmetrize3<N, T> operation_t;
    typedef symmetry_operation_dispatcher<operation_t> dispatcher_t;

private:
    const symmetry<N, T> &m_sym1; //!< Symmetry group
    permutation<N> m_cperm; //!< Cyclic %permutation to be added to %symmetry
    permutation<N> m_pperm; //!< Pairwise %permutation to be added to %symmetry
    bool m_symm; //!< Symmetric/anti-symmetric flag

public:
    /**	\brief Initializes the operation
		\param sym1 Symmetry container.
		\param cperm Cyclic permutation to be added (order = 3).
		\param pperm Pairwise permutation to be added (order = 2).
		\param symm Symmetric (true)/anti-symmetric (false)
     **/
    so_symmetrize3(const symmetry<N, T> &sym1, const permutation<N> &cperm,
            const permutation<N> &pperm, bool symm) :
                m_sym1(sym1), m_cperm(cperm), m_pperm(pperm), m_symm(symm) { }

    /**	\brief Performs the operation
		\param sym2 Destination %symmetry container.
     **/
    void perform(symmetry<N, T> &sym2);

private:
    so_symmetrize3(const so_symmetrize3<N, T>&);
    const so_symmetrize3<N, T> &operator=(const so_symmetrize3<N, T>&);
};

template<size_t N, typename T>
class symmetry_operation_params< so_symmetrize3<N, T> > :
public symmetry_operation_params_i {

public:
    const symmetry_element_set<N, T> &grp1; //!< Symmetry group 1
    permutation<N> cperm; //!< Cyclic permutation
    permutation<N> pperm; //!< Pairwise permutation
    bool symm; //!< Symmetrize/anti-symmetrize
    symmetry_element_set<N, T> &grp2; //!< Symmetry group 2 (output)

public:
    symmetry_operation_params(
            const symmetry_element_set<N, T> &grp1_,
            const permutation<N> &cperm_, const permutation<N> &pperm_,
            bool symm_, symmetry_element_set<N, T> &grp2_) :

                grp1(grp1_), cperm(cperm_), pperm(pperm_), symm(symm_),
                grp2(grp2_) { }

    virtual ~symmetry_operation_params() { }
};


} // namespace libtensor

#include "so_symmetrize3_handlers.h"

#endif // LIBTENSOR_SO_SYMMETRIZE3_H
