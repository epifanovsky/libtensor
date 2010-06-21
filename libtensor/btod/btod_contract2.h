#ifndef LIBTENSOR_BTOD_CONTRACT2_H
#define LIBTENSOR_BTOD_CONTRACT2_H

#include <list>
#include <map>
#include "../defs.h"
#include "../exception.h"
#include "../timings.h"
#include "../core/abs_index.h"
#include "../core/block_tensor_i.h"
#include "../core/block_tensor_ctrl.h"
#include "../core/orbit.h"
#include "../core/orbit_list.h"
#include "../core/sequence.h"
#include "../tod/contraction2.h"
#include "additive_btod.h"
#include "../not_implemented.h"
#include "bad_block_index_space.h"

namespace libtensor {


/**	\brief Contraction of two block tensors

	\ingroup libtensor_btod
 **/
template<size_t N, size_t M, size_t K>
class btod_contract2 :
	public additive_btod<N + M>,
	public timings< btod_contract2<N, M, K> > {

public:
	static const char *k_clazz; //!< Class name

private:
	static const size_t k_ordera = N + K; //!< Order of first argument (a)
	static const size_t k_orderb = M + K; //!< Order of second argument (b)
	static const size_t k_orderc = N + M; //!< Order of result (c)
	static const size_t k_totidx = N + M + K; //!< Total number of indexes
	static const size_t k_maxconn = 2 * k_totidx; //!< Index connections

private:
	typedef struct block_contr {
	public:
		size_t m_absidxa;
		size_t m_absidxb;
		double m_c;
		permutation<k_ordera> m_perma;
		permutation<k_orderb> m_permb;

	public:
		block_contr(size_t aia, size_t aib, double c,
			const permutation<k_ordera> &perma,
			const permutation<k_orderb> &permb)
		: m_absidxa(aia), m_absidxb(aib), m_c(c), m_perma(perma),
			m_permb(permb)
		{ }
		bool is_same_perm(const transf<k_ordera, double> &tra,
			const transf<k_orderb, double> &trb) {

			return m_perma.equals(tra.get_perm()) &&
				m_permb.equals(trb.get_perm());
		}
	} block_contr_t;
	typedef std::list<block_contr_t> block_contr_list_t;
	typedef std::map<size_t, block_contr_list_t*> schedule_t;

private:
	contraction2<N, M, K> m_contr; //!< Contraction
	block_tensor_i<k_ordera, double> &m_bta; //!< First argument (A)
	block_tensor_i<k_orderb, double> &m_btb; //!< Second argument (B)
	block_index_space<k_orderc> m_bis; //!< Block %index space of the result
	symmetry<k_orderc, double> m_sym; //!< Symmetry of the result
	dimensions<k_ordera> m_bidimsa; //!< Block %index dims of A
	dimensions<k_orderb> m_bidimsb; //!< Block %index dims of B
	dimensions<k_orderc> m_bidimsc; //!< Block %index dims of the result
	schedule_t m_contr_sch; //!< Contraction schedule
	assignment_schedule<k_orderc, double> m_sch; //!< Assignment schedule

public:
	//!	\name Construction and destruction
	//@{

	/**	\brief Initializes the contraction operation
		\param contr Contraction.
		\param bta Block %tensor A (first argument).
		\param btb Block %tensor B (second argument).
	 **/
	btod_contract2(const contraction2<N, M, K> &contr,
		block_tensor_i<k_ordera, double> &bta,
		block_tensor_i<k_orderb, double> &btb);

	/**	\brief Virtual destructor
	 **/
	virtual ~btod_contract2();

	//@}

	//!	\name Implementation of
	//		libtensor::direct_block_tensor_operation<N + M, double>
	//@{

	virtual const block_index_space<N + M> &get_bis() const {
		return m_bis;
	}

	virtual const symmetry<N + M, double> &get_symmetry() const {
		return m_sym;
	}

	virtual const assignment_schedule<N + M, double> &get_schedule() const {
		return m_sch;
	}

	//@}

	using additive_btod<N + M>::perform;

protected:
	virtual void compute_block(tensor_i<N + M, double> &blk,
		const index<N + M> &i);
	virtual void compute_block(tensor_i<N + M, double> &blk,
		const index<N + M> &i, const transf<N + M, double> &tr,
		double c);

private:
	static block_index_space<N + M> make_bis(
		const contraction2<N, M, K> &contr,
		block_tensor_i<k_ordera, double> &bta,
		block_tensor_i<k_orderb, double> &btb);
	void make_symmetry();
	void make_schedule();

	/**	\brief For an orbit in a and b, make a list of blocks in c
	 **/
	void make_schedule(const orbit<k_ordera, double> &oa,
		const orbit<k_orderb, double> &ob,
		const orbit_list<k_orderc, double> &olc);

	void clear_schedule(schedule_t &sch);

	void contract_block(
		block_contr_list_t &lst, const index<k_orderc> &idxc,
		block_tensor_ctrl<k_ordera, double> &ctrla,
		block_tensor_ctrl<k_orderb, double> &ctrlb,
		tensor_i<k_orderc, double> &blkc,
		const transf<k_orderc, double> &trc,
		bool zero, double c);

private:
	btod_contract2(const btod_contract2<N, M, K>&);
	btod_contract2<N, M, K> &operator=(const btod_contract2<N, M, K>&);

};


} // namespace libtensor


#ifndef LIBTENSOR_INSTANTIATE_TEMPLATES
#include "btod_contract2_impl.h"
#endif // LIBTENSOR_INSTANTIATE_TEMPLATES


#endif // LIBTENSOR_BTOD_CONTRACT2_H
