#ifndef LIBTENSOR_LOOP_LIST_MUL_H
#define LIBTENSOR_LOOP_LIST_MUL_H

#include <list>
#include "../timings.h"
#include "loop_list_base.h"

namespace libtensor {


/**	\brief Operates nested loops on two arrays with multiplication and
		accumulation as the kernel (c += a * b)

	\ingroup libtensor_tod
 **/
class loop_list_mul :
	public loop_list_base<2, 1, loop_list_mul>,
	public timings<loop_list_mul> {

public:
	static const char *k_clazz; //!< Class name

private:
	struct {
		double m_d;
		size_t m_n;
		size_t m_stepa, m_stepb, m_stepc;
	} m_generic;

	//!	c = a_i b_i
	struct {
		double m_d;
		size_t m_n, m_stepa, m_stepb;
	} m_ddot;

	//!	c = a_ij b_ji
	struct {
		double m_d;
		size_t m_ni, m_nj, m_lda, m_ldb;
	} m_ddot_trp;

	//!	c_i = a_i b
	struct {
		double m_d;
		size_t m_n, m_stepc;
	} m_daxpy_a;

	//!	c_i = a b_i
	struct {
		double m_d;
		size_t m_n, m_stepc;
	} m_daxpy_b;

	//!	c_i = a_ip b_p
	struct {
		double m_d;
		size_t m_rows, m_cols, m_stepb, m_lda;
	} m_dgemv_n_a;

	//!	c_i = a_pi b_p
	struct {
		double m_d;
		size_t m_rows, m_cols, m_stepb, m_lda, m_stepc;
	} m_dgemv_t_a;

	//!	c_i = a_p b_ip
	struct {
		double m_d;
		size_t m_rows, m_cols, m_stepa, m_ldb;
	} m_dgemv_n_b;

	//!	c_i = a_p b_pi
	struct {
		double m_d;
		size_t m_rows, m_cols, m_stepa, m_ldb, m_stepc;
	} m_dgemv_t_b;

	//!	c_ij = a_ip b_pj
	struct {
		double m_d;
		size_t m_rowsa, m_colsb, m_colsa, m_lda, m_ldb, m_ldc;
	} m_dgemm_nn_ab;

	//!	c_ij = a_ip b_jp
	struct {
		double m_d;
		size_t m_rowsa, m_colsb, m_colsa, m_lda, m_ldb, m_ldc;
	} m_dgemm_nt_ab;

	//!	c_ij = a_pi b_pj
	struct {
		double m_d;
		size_t m_rowsa, m_colsb, m_colsa, m_lda, m_ldb, m_ldc;
	} m_dgemm_tn_ab;

	//!	c_ij = a_pi b_jp
	struct {
		double m_d;
		size_t m_rowsa, m_colsb, m_colsa, m_lda, m_ldb, m_ldc;
	} m_dgemm_tt_ab;

	//!	c_ij = a_pj b_ip
	struct {
		double m_d;
		size_t m_rowsb, m_colsa, m_colsb, m_ldb, m_lda, m_ldc;
	} m_dgemm_nn_ba;

	//!	c_ij = a_jp b_ip
	struct {
		double m_d;
		size_t m_rowsb, m_colsa, m_colsb, m_ldb, m_lda, m_ldc;
	} m_dgemm_nt_ba;

	//!	c_ij = a_pj b_pi
	struct {
		double m_d;
		size_t m_rowsb, m_colsa, m_colsb, m_ldb, m_lda, m_ldc;
	} m_dgemm_tn_ba;

	//!	c_ij = a_pj b_ip
	struct {
		double m_d;
		size_t m_rowsb, m_colsa, m_colsb, m_ldb, m_lda, m_ldc;
	} m_dgemm_tt_ba;

	const char *m_kernelname;

protected:
	void run_loop(list_t &loop, registers &r, double c);

private:
	void match_l1(list_t &loop, double d);
	void match_ddot_l2(list_t &loop, double d, size_t w1, size_t k1);
	void match_daxpy_a_l2(list_t &loop, double d, size_t w1, size_t k1);
	void match_daxpy_b_l2(list_t &loop, double d, size_t w1, size_t k1);
	void match_dgemv_n_a_l3(list_t &loop, double d, size_t w1, size_t w2,
		size_t k1w1);
	void match_dgemv_n_b_l3(list_t &loop, double d, size_t w1, size_t w2,
		size_t k1w1);
	void match_dgemv_t_a1_l3(list_t &loop, double d, size_t w1, size_t w2,
		size_t k1, size_t k2w1);
	void match_dgemv_t_a2_l3(list_t &loop, double d, size_t w1, size_t w2,
		size_t k2w1, size_t k3);
	void match_dgemv_t_b1_l3(list_t &loop, double d, size_t w1, size_t w2,
		size_t k1, size_t k2w1);
	void match_dgemv_t_b2_l3(list_t &loop, double d, size_t w1, size_t w2,
		size_t k2w1, size_t k3);
	void fn_generic(registers &r) const;
	void fn_ddot(registers &r) const;
	void fn_ddot_trp(registers &r) const;
	void fn_daxpy_a(registers &r) const;
	void fn_daxpy_b(registers &r) const;
	void fn_dgemv_n_a(registers &r) const;
	void fn_dgemv_t_a(registers &r) const;
	void fn_dgemv_n_b(registers &r) const;
	void fn_dgemv_t_b(registers &r) const;
	void fn_dgemm_nn_ab(registers &r) const;
	void fn_dgemm_nt_ab(registers &r) const;
	void fn_dgemm_tn_ab(registers &r) const;
	void fn_dgemm_tt_ab(registers &r) const;
	void fn_dgemm_nn_ba(registers &r) const;
	void fn_dgemm_nt_ba(registers &r) const;
	void fn_dgemm_tn_ba(registers &r) const;
	void fn_dgemm_tt_ba(registers &r) const;

};


} // namespace libtensor

#endif // LIBTENSOR_LOOP_LIST_MUL_H
