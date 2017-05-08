/*******************************************************************\

Module: Goto Programs with Functions

Author: Daniel Kroening

Date: June 2003

\*******************************************************************/

#ifndef CPROVER_GOTO_FUNCTIONS_H
#define CPROVER_GOTO_FUNCTIONS_H

#define Forall_goto_functions(it, functions) \
  for(goto_functionst::function_mapt::iterator it=(functions).function_map.begin(); \
      it!=(functions).function_map.end(); it++)

#define forall_goto_functions(it, functions) \
  for(goto_functionst::function_mapt::const_iterator it=(functions).function_map.begin(); \
      it!=(functions).function_map.end(); it++)

#include <iostream>

#include <std_types.h>

#include "goto_program.h"

class goto_functiont
{
public:
  goto_programt body;
  code_typet type;
  bool body_available;

  // The set of functions that have been inlined into this one. Necessary to
  // make symex renaming work.
  std::set<std::string> inlined_funcs;

  bool is_inlined() const
  {
    return type.inlined();
  }

  goto_functiont():body_available(false)
  {
  }

  void clear()
  {
    body.clear();
    type.clear();
    body_available=false;
    inlined_funcs.clear();
  }
};

class goto_functionst
{
public:
  typedef std::map<irep_idt, goto_functiont> function_mapt;
  function_mapt function_map;

  ~goto_functionst() { }
  void clear()
  {
    function_map.clear();
  }

  void output(
    const namespacet &ns,
    std::ostream &out) const;

  void compute_location_numbers();
  void compute_loop_numbers();
  void compute_target_numbers();
  void compute_incoming_edges();

  void update()
  {
    compute_incoming_edges();
    compute_target_numbers();
    compute_location_numbers();
  }

  irep_idt main_id() const
  {
    return "main";
  }

  void swap(goto_functionst &other)
  {
    function_map.swap(other.function_map);
  }
};

#endif
