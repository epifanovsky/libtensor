#ifndef LIBTENSOR_SO_APPLY_H
#define LIBTENSOR_SO_APPLY_H

#include <libtensor/core/symmetry.h>
#include "symmetry_operation_base.h"
#include "symmetry_operation_params.h"

namespace libtensor {

template<size_t N, typename T>
class so_apply;

template<size_t N, typename T>
class symmetry_operation_params< so_apply<N, T> >;


/**	\brief Computes the %symmetry of a tensor subjected to some functor
	\tparam N Symmetry cardinality (%tensor order).
	\tparam T Tensor element type.

	\ingroup libtensor_symmetry
 **/
template<size_t N, typename T>
class so_apply : public symmetry_operation_base< so_apply<N, T> > {
private:
    typedef so_apply<N, T> operation_t;
    typedef symmetry_operation_dispatcher<operation_t> dispatcher_t;

private:
    const symmetry<N, T> &m_sym1; //!< Symmetry container (A)
    permutation<N> m_perm1; //!< Permutation of the %tensor
    bool m_keep_zero; //!< Functor maps 0 to 0
    bool m_is_asym; //!< Functor is asymmetric
    bool m_sign; //!< Functor is symmetric or anti-symmetric

public:
    /**	\brief Initializes the operation
		\param sym %Symmetry container (A).
		\param perm Permutation of the %tensor.
		\param is_asym Functor is asymmetric.
		\param sign Functor is symmetric or anti-symmetric
			(ignored if is_asym is true).
     **/
    so_apply(const symmetry<N, T> &sym1, const permutation<N> &perm1,
            bool keep_zero, bool is_asym, bool sign) :
                m_sym1(sym1), m_perm1(perm1), m_keep_zero(keep_zero),
                m_is_asym(is_asym), m_sign(sign)
    { }

    /**	\brief Performs the operation
		\param sym Destination %symmetry container.
     **/
    void perform(symmetry<N, T> &sym);

private:
    so_apply(const so_apply<N, T>&);
    const so_apply<N, T> &operator=(const so_apply<N, T>&);
};

template<size_t N, typename T>
class symmetry_operation_params< so_apply<N, T> > :
    public symmetry_operation_params_i {

public:
    const symmetry_element_set<N, T> &grp1; //!< Symmetry group 1
    permutation<N> perm1; //!< Permutation 1
    bool keep_zero; //!< Functor maps 0 to 0
    bool is_asym; //!< Functor is asymmetric
    bool sign; //!< Functor is symmetric or anti-symmetric
    symmetry_element_set<N, T> &grp2; //!< Symmetry group 2 (output)

public:
    symmetry_operation_params(
            const symmetry_element_set<N, T> &grp1_,
            const permutation<N> &perm1_,
            bool keep_zero_, bool is_asym_, bool sign_,
            symmetry_element_set<N, T> &grp2_) :

                grp1(grp1_), perm1(perm1_), keep_zero(keep_zero_),
                is_asym(is_asym_), sign(sign_), grp2(grp2_) { }

    virtual ~symmetry_operation_params() { }
};


} // namespace libtensor

#include "so_apply_handlers.h"

#endif // LIBTENSOR_SO_APPLY_H

