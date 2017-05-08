/*******************************************************************\

Module: Goto Programs with Functions

Author: Daniel Kroening

Date: June 2003

\*******************************************************************/

#ifndef CPROVER_GOTO_CONVERT_FUNCTIONS_H
#define CPROVER_GOTO_CONVERT_FUNCTIONS_H

#include "goto_functions.h"
#include "goto_convert_class.h"

// just convert it all
void goto_convert(
  contextt &context,
  optionst &options,
  goto_functionst &functions,
  message_handlert &message_handler);

class goto_convert_functionst:public goto_convertt
{
public:
  typedef std::map<irep_idt, std::set<irep_idt> > typename_mapt;
  typedef std::set<irep_idt> typename_sett;
  typedef std::list<symbolt*> symbol_listt;

  void goto_convert();
  void convert_function(symbolt &symbol);
  void convert_function(const irep_idt &identifier);
  void thrash_type_symbols(void);
  void collect_type(const irept &type, typename_sett &set);
  void collect_expr(const irept &expr, typename_sett &set);
  void rename_types(irept &type, const symbolt &cur_name_sym,
                    const irep_idt &sname);
  void rename_exprs(irept &expr, const symbolt &cur_name_sym,
                    const irep_idt &sname);
  void wallop_type(irep_idt name, typename_mapt &typenames,
                   const irep_idt &sname);

  goto_convert_functionst(
    contextt &_context,
    optionst &_options,
    goto_functionst &_functions,
    message_handlert &_message_handler);

  virtual ~goto_convert_functionst();

protected:
  goto_functionst &functions;

  static bool hide(const goto_programt &goto_program);

  //
  // function calls
  //
  void add_return(
    goto_functiont &f,
    const locationt &location);

private:
  bool inlining;

};

#endif
