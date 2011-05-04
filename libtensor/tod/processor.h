#ifndef LIBTENSOR_PROCESSOR_H
#define LIBTENSOR_PROCESSOR_H

#include "../defs.h"
#include "../exception.h"

namespace libtensor {

template<typename List, typename Registers> class processor;

/**	\brief Processor operation interface

	\ingroup libtensor_tod
 **/
template<typename List, typename Registers>
class processor_op_i {
public:
	typedef processor<List, Registers> processor_t;
	typedef Registers registers_t;

public:
	virtual void exec(processor_t &proc, registers_t &regs)
		throw(exception) = 0;
};

/**	\brief Processes a recursive sequence of operations

	\ingroup libtensor_tod
 **/
template<typename List, typename Registers>
class processor {
public:
	typedef processor_op_i<List, Registers> operation_t;
	typedef List list_t;
	typedef typename List::const_iterator iterator_t;
	typedef Registers registers_t;

private:
	const list_t &m_list; //!< List of operations
	iterator_t m_iter; //!< Operation iterator
	registers_t &m_regs; //!< Registers

public:
	processor(const list_t &list, registers_t &regs);
	void process_next() throw(exception);
};

template<typename List, typename Registers>
processor<List, Registers>::processor(const list_t &list, registers_t &regs) :
	m_list(list), m_iter(list.begin()), m_regs(regs) {
}

template<typename List, typename Registers>
void processor<List, Registers>::process_next() throw(exception) {
	if(m_iter == m_list.end()) return;

	if(m_iter->op() == NULL) {
		throw_exc("processor<List, Registers>",
			"process_next()",
			"NULL pointer exception: m_iter->op()");
	}

	iterator_t iter = m_iter;
	m_iter++;
	operation_t *op = iter->op();
	op->exec(*this, m_regs);
	m_iter = iter;
}

} // namespace libtensor

#endif // LIBTENSOR_PROCESSOR_H
