#ifndef LIBTENSOR_LOOP_LIST_APPLY_H
#define LIBTENSOR_LOOP_LIST_APPLY_H

#include <list>
#include "loop_list_base.h"

namespace libtensor {

/** \brief Supplementary class used in list_loop_apply to perform the
		inner loop computation.
 **/
template<typename Functor>
struct apply_base {

	typedef Functor functor_t;

	/** \brief \f$ c_i = b0 * fn(b1 * a_i) \f$
		\param ni Number of elements i.
		\param fn Function pointer fn.
		\param a Input data array a.
		\param sia Step of i in a.
		\param b0 Scalar b0.
		\param b1 Scalar b1.
		\param c Result data array c.
		\param sic Step of i in c.
	 **/
	static void set(size_t ni, functor_t &fn,
			const double *a, size_t sia, double b0, double b1,
			double *c, size_t sic);

	/** \brief \f$ c_i = c_i + b0 * fn(b1 * a_i) \f$
		\param ni Number of elements i.
		\param fn Function pointer fn.
		\param a Input data array a.
		\param sia Step of i in a.
		\param b0 Scalar b0.
		\param b1 Scalar b1.
		\param c Result data array c.
		\param sic Step of i in c.
	 **/
	static void add(size_t ni, functor_t &fn,
			const double *a, size_t sia, double b0, double b1,
			double *c, size_t sic);
};

/**	\brief Operates nested loops on two arrays with function pointer
		as the kernel (b = c f(c0 a) or b = b + c f(c0 a))

	\ingroup libtensor_tod
 **/
template<typename Functor>
class loop_list_apply :
	public loop_list_base<1, 1, loop_list_apply<Functor> > {
public:
	static const char *k_clazz; //!< Class name

	typedef Functor functor_t;
	typedef typename loop_list_base<1, 1, loop_list_apply<functor_t> >::list_t list_t;
	typedef typename loop_list_base<1, 1, loop_list_apply<functor_t> >::registers registers;
	typedef typename loop_list_base<1, 1, loop_list_apply<functor_t> >::iterator_t iterator_t;

private:
	struct {
		double m_c0, m_c1;
		size_t m_n;
		size_t m_stepa;
		size_t m_stepb;
		functor_t *m_fn;
	} m_apply;

protected:
	void run_loop(list_t &loop, registers &r,
			functor_t &fn, double c0, double c1, bool do_add);

private:
	void install_kernel(list_t &loop,
			functor_t &fn, double c0, double c1, bool do_add);
	void fn_apply_add(registers &r) const;
	void fn_apply_set(registers &r) const;
};


} // namespace libtensor

#include "loop_list_apply_impl.h"

#endif // LIBTENSOR_LOOP_LIST_APPLY_H
