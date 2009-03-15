#ifndef LIBTENSOR_BUFFERING_H
#define LIBTENSOR_BUFFERING_H

#include "defs.h"
#include "exception.h"

namespace libtensor {

/**	\brief Implements the buffering feature in direct tensors

	\ingroup libtensor
**/
class buffering {
private:
	bool m_buffering; //!< Indicates whether buffering is enabled;

public:
	//!	\name Construction and destruction
	//@{
	buffering();
	virtual ~buffering();
	//@}

	//!	\name Buffering
	//@{
	void enable_buffering();
	void disable_buffering();
	bool is_buffering_enabled();
	//@}
};

inline buffering::buffering() : m_buffering(false) {
}

inline buffering::~buffering() {
}

inline void buffering::enable_buffering() {
	m_buffering = true;
}

inline void buffering::disable_buffering() {
	m_buffering = false;
}

inline bool buffering::is_buffering_enabled() {
	return m_buffering;
}

} // namespace libtensor

#endif // LIBTENSOR_BUFFERING_H

