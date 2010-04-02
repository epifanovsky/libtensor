#ifndef LIBTENSOR_LOOP_LIST_COPY_H
#define LIBTENSOR_LOOP_LIST_COPY_H

#include <list>
#include "../defs.h"

namespace libtensor {


/**	\brief Operates nested loops on two arrays with assignment
		as the kernel (copy)

	\ingroup libtensor_tod
 **/
class loop_list_copy {
public:
	static const char *k_clazz; //!< Class name

protected:
	/**	\brief Structure keeps track of the current location in both
			arrays
	 **/
	struct registers {
		const double *m_ptra; //!< Position in the first (source) array
		double *m_ptrb; //!< Position in the second (destination) array
#ifdef LIBTENSOR_DEBUG
		const double *m_ptra_end; //!< End of source array
		double *m_ptrb_end; //!< End of destination array
#endif // LIBTENSOR_DEBUG
	};

	struct node;
	typedef std::list<node> list_t; //!< List of nested loops (type)
	typedef std::list<node>::iterator iterator_t; //!< List iterator type

	/**	\brief Node on the list of nested loops
	 **/
	struct node {
	public:
		typedef void (loop_list_copy::*fnptr_t)(registers&) const;

	private:
		size_t m_weight; //!< Number of iterations in the loop
		size_t m_stepa; //!< Increment in the first (source) array
		size_t m_stepb; //!< Increment in the second (destination) array
		fnptr_t m_fn; //!< Function

	public:
		/**	\brief Default constructor
		 **/
		node() : m_weight(0), m_stepa(0), m_stepb(0), m_fn(0) { }

		/**	\brief Initializing constructor
		 **/
		node(size_t weight, size_t stepa, size_t stepb) :
			m_weight(weight), m_stepa(stepa), m_stepb(stepb),
			m_fn(0) { }

		size_t &weight() { return m_weight; }
		size_t &stepa() { return m_stepa; }
		size_t &stepb() { return m_stepb; }
		fnptr_t &fn() { return m_fn; }
	};

	struct {
		double m_k;
		size_t m_n;
		size_t m_stepa;
		size_t m_stepb;
	} m_copy;

protected:
	void run_loop(list_t &loop, registers &regs, double c);

private:
	void exec(iterator_t &i, iterator_t &iend, registers &r);
	void fn_loop(iterator_t &i, iterator_t &iend, registers &r);
	void fn_copy(registers &r) const;
};


} // namespace libtensor

#endif // LIBTENSOR_LOOP_LIST_COPY_H
