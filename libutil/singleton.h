#ifndef LIBUTIL_SINGLETON_H
#define LIBUTIL_SINGLETON_H

namespace libutil {


/** \brief Meyers %singleton template

    The %singleton pattern only allows one instance of a class. This
    template can be used to turn any class into a %singleton.

    To become a %singleton, a class has to:
     - derive from this template with the class as the template parameter;
     - friend the template with the class as the template parameter;
     - implement a protected default constructor.

    The static get_instance() method can be used to obtain the %singleton
    instance.

    Example:
    \code
    class c : public singleton<c> {
        friend class singleton<c>;
    protected:
        c() {}

    public:
        void do_something();
        // ... other members of the class ...
    }

    int main() {
        // Returns the instance of c
        c::get_instance().do_something();
    }
    \endcode

    \ingroup libutil
 **/
template<typename T>
class singleton {
public:
    /** \brief Returns the instance of the underlying class
     **/
    static T &get_instance();

    /** \brief Virtual destructor
     **/
    virtual ~singleton();

};


template<typename T>
inline T &singleton<T>::get_instance() {

    static T instance;
    return instance;
}


template<typename T>
inline singleton<T>::~singleton() {

}


} // namespace libutil

#endif // LIBUTIL_SINGLETON_H

