#ifndef __TENSOR_EXCEPTION_H
#define __TENSOR_EXCEPTION_H

#include <cstring>
#include <exception>

namespace tensor {

class exception : public std::exception {
private:
	char m_msg[1024];

public:
	exception(const char *msg) throw();
	virtual ~exception() throw();
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

}

#endif // __TENSOR_EXCEPTION_H

