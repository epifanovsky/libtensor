#ifndef LIBTENSOR_PERMUTATION_GROUP_H
#define LIBTENSOR_PERMUTATION_GROUP_H

#include <algorithm>
#include <list>
#include "se_perm.h"
#include "symmetry_element_set_adapter.h"


namespace libtensor {


/**	\brief Stores the reduced representation of a generalized %permutation
        group
	\tparam N Tensor order.
	\tparam T Tensor element type.

	This class implements a modified version of the %permutation group
	container and procedures described in
	    M. Jerrum, J. Algorithms 7 (1986), 60-78.
	The modifications are such that each group element has a scalar
	transformation attached attached to it. The requirement to the
	scalar transformation S is that it is the generator of a cyclic
	group of order m (i.e. \f$ S^m = 1 \f$), and that m is a divisor
	of the order n of the permutation P it is attached to
	(\f$ P^n = 1 \f$). Both requirements are currently not checked.
	They are however checked for class \c se_perm<N, T>.

	\ingroup libtensor_symmetry
 **/
template<size_t N, typename T>
class permutation_group {
public:
    static const char *k_clazz; //!< Class name

private:
    typedef permutation<N> perm_t;
    typedef scalar_transf<T> transf_t;
    typedef std::pair<perm_t, transf_t> gen_perm_t;

    //!	Stores one labeled branching
    struct branching {
        gen_perm_t m_sigma[N]; //!< Edge labels (permutation + n)
        gen_perm_t m_tau[N]; //!< Vertex labels (permutation + n)
        size_t m_edges[N]; //!< Edge sources
        branching() {
            for(register size_t i = 0; i < N; i++) m_edges[i] = N;
        }
        void reset() {
            for(register size_t i = 0; i < N; i++) {
                m_edges[i] = N;
                m_sigma[i].first.reset();
                m_tau[i].first.reset();
                m_sigma[i].second.reset();
                m_tau[i].second.reset();
            }
        }
    };

private:
    typedef se_perm<N, T> se_perm_t;

    typedef std::list<gen_perm_t> perm_list_t;
    typedef std::vector<gen_perm_t> perm_vec_t;

private:
    branching m_br; //!< Branching

public:
    //!	\name Construction and destruction
    //@{

    /**	\brief Creates the C1 group
     **/
    permutation_group() { }

    /**	\brief Creates a %permutation group from a generating set
		\param set Generating set.
     **/
    permutation_group(
            const symmetry_element_set_adapter<N, T, se_perm_t> &set);

    /**	\brief Destroys the object
     **/
    ~permutation_group() { }

    //@}


    //!	\name Manipulations
    //@{

    /**	\brief Augments the group with an %orbit represented by a
			%permutation.
		\param tr Scalar transformation
		\param perm Permutation.

		Does nothing if the subgroup with the same sign already contains
		the %orbit. Throws bad_symmetry if the subgroup with the
		opposite sign contains the %orbit.
     **/
    void add_orbit(const scalar_transf<T> &tr, const permutation<N> &perm);


    /**	\brief Tests the membership of a %permutation in the group
		\param tr Scalar transformation.
		\param perm Permutation.
     **/
    bool is_member(const scalar_transf<T> &tr,
            const permutation<N> &perm) const;


    /**	\brief Converts the %permutation group to a generating set
			using the standard format
     **/
    void convert(symmetry_element_set<N, T> &set) const;


    /**	\brief Generates a subgroup of all permutations which
			stabilize unmasked elements. The %mask must have M
			masked elements for the operation to succeed.
     **/
    template<size_t M>
    void project_down(const mask<N> &msk, permutation_group<M, T> &g2);


    /** \brief Generates a subgroup that stabilize the set of masked indexes.
	 	\param msk Set of elements to be stabilized.
		\param g2 Resulting subgroup.

		The resulting subgroup will contain all permutations that map the set
		of masked indexes onto itself.
     **/
    void stabilize(const mask<N> &msk, permutation_group<N, T> &g2);


    /** \brief Generates a subgroup that set-wise stabilizes groups of indexes.
	 	\param seq Sequences to specify the indexes that are stabilized
		\param g2 Resulting subgroup.

        The given sequence specifies the sets of indexes which are stabilized:
        indexes for which \c seq has value 0 are not stabilized and those for
        which it has value 1, 2, ... belong to the first, second, ... set of
        stabilized indexes. The values for the sets have to be consecutive
        starting at 1. The resulting subgroup will contain all permutations
        that map each of the sets of indexes onto itself or onto another set
        of the same size.
     **/
    void stabilize(const sequence<N, size_t> &seq, permutation_group<N, T> &g2);


    /** \brief Permute the indexes in the permutation group.
     **/
    void permute(const permutation<N> &perm);

    //@}

private:
    /**	\brief Computes the non-trivial path from node i to node j
			(j > i). Returns the length of the path or 0 if such
			path doesn't exist
     **/
    size_t get_path(const branching &br, size_t i, size_t j,
            size_t (&path)[N]) const;


    /**	\brief Tests the membership of a %permutation in G_{i-1}
			(or G for i==0)

	    The function modifies the given scalar transformation
	    T_r to \f$ T_r T^{-1}_{i-1} \f$ where T_{i-1} is the
	    transformation assigned to the permutation in G_{i-1}.
     **/
    bool is_member(const branching &br, size_t i, scalar_transf<T> &tr,
            const permutation<N> &perm) const;


    /**	\brief Computes a branching using a generating set; returns
			the generating set of G_{i-1}
     **/
    void make_branching(branching &br, size_t i, const perm_list_t &gs,
            perm_list_t &gs2);


    void make_genset(const branching &br, perm_list_t &gs) const;


    void permute_branching(branching &br, const permutation<N> &perm);


    /**	\brief Computes a generating set for the subgroup that stabilizes
			a set given by msk
		\param br Branching representing the group
		\param msk Sequence specifying the sets to stabilize
		\param gs Generating set for the subgroup

		The mask indicates set which is to be stabilized as follows
		- Elements with identical numbers can be permuted
		- Sets of elements with numbers other than zero can be permuted as a
		  whole

     **/
    void make_setstabilizer(const branching &br,
            const sequence<N, size_t> &msk, perm_list_t &gs);
};


template<size_t N, typename T>
inline
bool permutation_group<N, T>::is_member(const scalar_transf<T> &tr,
        const permutation<N> &perm) const {

    if(perm.is_identity()) {
        return tr.is_identity();
    }
    scalar_transf<T> trx(tr);
    if (is_member(m_br, 0, trx, perm)) {
        return trx.is_identity();
    }

    return false;
}


} // namespace libtensor

#endif // LIBTENSOR_PERMUTATION_GROUP_H
