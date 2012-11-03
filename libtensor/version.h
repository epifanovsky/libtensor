#ifndef LIBTENSOR_VERSION_H
#define LIBTENSOR_VERSION_H

#include <list>
#include <string>
#include <libutil/singleton.h>

namespace libtensor {


/** \brief Version of the %tensor library

    The %version of the library is specified using the %version number and
    status. The number consists of a major number and a minor number. The
    status string describes the release status.

    For example, %version 2.0-alpha2 has major number 2, minor number 0,
    and status "alpha2" meaning the second alpha release.

    \ingroup libtensor
 **/
class version : public libutil::singleton<version> {
    friend class libutil::singleton<version>;

private:
    static const unsigned k_major = 2; //!< Major %version number
    static const unsigned k_minor = 2; //!< Minor %version number
    static const char *k_status; //!< Version status
    static const char *k_authors[]; //!< List of authors

private:
    std::string m_status; //!< Version status
    std::string m_string; //!< Version string
    std::list<std::string> m_authors; //!< List of authors

protected:
    version();

    const std::string &get_status_impl() const {
        return m_status;
    }

    const std::string &get_string_impl() const {
        return m_string;
    }

    const std::list<std::string> &get_authors_impl() const {
        return m_authors;
    }

public:
    /**    \brief Returns the major %version number
     **/
    static unsigned get_major() {
        return k_major;
    }

    /**    \brief Returns the minor %version number
     **/
    static unsigned get_minor() {
        return k_minor;
    }

    /**    \brief Returns the %version status
     **/
    static const std::string &get_status() {
        return version::get_instance().get_status_impl();
    }

    /**    \brief Returns the string that corresponds to the %version
     **/
    static const std::string &get_string() {
        return version::get_instance().get_string_impl();
    }

    /**    \brief Returns the list of authors
     **/
    static const std::list<std::string> &get_authors() {
        return version::get_instance().get_authors_impl();
    }

};


} // namespace libtensor

#endif // LIBTENSOR_VERSION_H
