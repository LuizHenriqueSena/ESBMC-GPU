/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_LANGUAGE_H
#define CPROVER_LANGUAGE_H

#include <set>
#include <stdio.h>
#include "context.h"
#include "namespace.h"
#include "message.h"

class languaget
{
public:
  // parse file

  virtual bool parse(
    const std::string &path,
    message_handlert &message_handler)=0;

  // add external dependencies of a given module to set

  virtual void dependencies()
  { }

  // add modules provided by currently parsed file to set

  virtual void modules_provided(std::set<std::string> &modules __attribute__((unused)))
  { }

  // final adjustments, e.g., initialization and call to main()
#if 1
  virtual bool final(
    contextt &context __attribute__((unused)),
    message_handlert &message_handler __attribute__((unused)))
  { return false; }
#endif
  // type check interfaces of currently parsed file

  virtual bool interfaces()
  { return false; }

  // type check a module in the currently parsed file

  virtual bool typecheck(
    contextt &context,
    const std::string &module,
    message_handlert &message_handler)=0;

  // language id / description

  virtual std::string id() const { return ""; }
  virtual std::string description() const { return ""; }

  // show parse tree

  virtual void show_parse(std::ostream &out)=0;

  // conversion of expressions

  virtual bool from_expr(
    const exprt &expr,
    std::string &code,
    const namespacet &ns);

  virtual bool from_type(
    const typet &type,
    std::string &code,
    const namespacet &ns);

  virtual bool to_expr(
    const std::string &code,
    const std::string &module,
    exprt &expr,
    message_handlert &message_handler,
    const namespacet &ns)=0;

  virtual languaget *new_language()=0;

  // constructor / destructor

  languaget() { }
  virtual ~languaget() { }
};
#endif
