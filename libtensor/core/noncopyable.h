#ifndef LIBTENSOR_NONCOPYABLE_H
#define LIBTENSOR_NONCOPYABLE_H

namespace libtensor {


/** \brief Non-copyable concept

    This class declares the copy constructor and the assignment operator
    as private members. For any class to become non-copyable, it needs to be
    derived from this class. An attempt to create a copy of a non-copyable
    object will result in a compilation error.

    \ingroup libtensor_core
 **/
class noncopyable {
protected:
    /** \brief Allowed default constructor
     **/
    noncopyable() { }

    /** \brief Empty destructor
     **/
    ~noncopyable() { }

private:
    /** \brief Private copy constructor
     **/
    noncopyable(const noncopyable&);

    /** \brief Private assignment operator
     **/
    const noncopyable &operator=(const noncopyable&);

};


} // namespace libtensor

#endif // LIBTENSOR_NONCOPYABLE_H
