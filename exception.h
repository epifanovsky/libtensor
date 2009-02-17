#ifndef __LIBTENSOR_EXCEPTION_H
#define __LIBTENSOR_EXCEPTION_H

#include <cstring>
#include <exception>

namespace libtensor {

/**	\brief Base class for exceptions in the tensor library
**/
class exception : public std::exception {
private:
	char m_msg[1024]; //! Keeps the message associated with the exception

public:
	/**	\brief Creates the exception object with a given message
	**/
	exception(const char *msg) throw();

	/**	\brief Virtual destructor
	**/
	virtual ~exception() throw();

	/**	\brief Returns the cause of the exception (message)
	**/
	virtual const char *what() const throw();
};

inline exception::exception(const char *msg) throw() {
	if(msg == NULL) m_msg[0] = '\0';
	else strncpy(m_msg, msg, 1024);
}

inline exception::~exception() throw() {
}

inline const char *exception::what() const throw() {
	return m_msg;
}

} // namespace libtensor

#endif // __LIBTENSOR_EXCEPTION_H

