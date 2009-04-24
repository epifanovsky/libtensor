#ifndef LIBTENSOR_RC_PTR_H
#define LIBTENSOR_RC_PTR_H

#include <memory>

namespace libtensor {

/**	\brief Reference counting pointer

	This smart pointer keeps track of the references to an object
	so it gets deleted after no other object is referencing it. This
	greatly simplifies memory management for certain situations.

	\ingroup libtensor
**/
template<typename T>
class rc_ptr {
private:
	struct counter {
		T *m_p;
		size_t m_count;
		counter(T *p);
	} *m_cnt; //!< Reference counter

public:
	explicit rc_ptr(T *p = NULL);

	template<typename T1>
	explicit rc_ptr(std::auto_ptr<T1> &p);

	rc_ptr(const rc_ptr<T> &p);

	~rc_ptr();

	rc_ptr<T> &operator=(const rc_ptr<T> &p);
	T &operator*() const;
	T *operator->() const;

private:
	void acquire(counter *cnt);
	void release();
};

template<typename T>
inline rc_ptr<T>::rc_ptr(T *p) {
	if(p) m_cnt = new counter(p);
	else m_cnt = NULL;
}

template<typename T> template<typename T1>
inline rc_ptr<T>::rc_ptr(std::auto_ptr<T1> &p) {
	T *p1 = (T1*)p.release();
	if(p1) m_cnt = new counter(p1);
	else m_cnt = NULL;
}

template<typename T>
inline rc_ptr<T>::rc_ptr(const rc_ptr<T> &p) {
	acquire(p.m_cnt);
}

template<typename T>
inline rc_ptr<T>::~rc_ptr() {
	release();
}

template<typename T>
inline rc_ptr<T> &rc_ptr<T>::operator=(const rc_ptr<T> &p) {
	if(this != &p) {
		release(); acquire(p.m_cnt);
	}
	return *this;
}

template<typename T>
inline T &rc_ptr<T>::operator*() const {
	return *m_cnt->m_p;
}

template<typename T>
inline T *rc_ptr<T>::operator->() const {
	return m_cnt->m_p;
}

template<typename T>
void rc_ptr<T>::acquire(counter *cnt) {
	m_cnt = cnt;
	if(m_cnt) m_cnt->m_count++;
}

template<typename T>
void rc_ptr<T>::release() {
	if(m_cnt) {
		if(--m_cnt->m_count == 0) {
			delete m_cnt->m_p;
			delete m_cnt;
		}
		m_cnt = NULL;
	}
}

template<typename T>
inline rc_ptr<T>::counter::counter(T *p) : m_p(p), m_count(1) {
}

} // namespace libtensor

#endif // LIBTENSOR_RC_PTR_H

