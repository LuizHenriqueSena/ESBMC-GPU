#ifndef _GOTO_SYMEX_RENAMING_H_
#define _GOTO_SYMEX_RENAMING_H_

#include <irep2.h>

#include <stdint.h>
#include <string.h>

#include <string>
#include <stack>
#include <vector>
#include <set>

#include <boost/functional/hash.hpp>

#include <guard.h>
#include <expr_util.h>
#include <std_expr.h>
#include <i2string.h>

#include "crypto_hash.h"

namespace renaming {

  struct renaming_levelt
  {
  public:
    virtual void get_original_name(expr2tc &expr) const = 0;
    virtual void rename(expr2tc &expr) = 0;
    virtual void remove(const expr2tc &symbol)=0;

    virtual void get_ident_name(expr2tc &symbol) const=0;

    virtual ~renaming_levelt() { }
  protected:
    void get_original_name(expr2tc &expr, symbol2t::renaming_level lev) const;
  };

  // level 1 -- function frames
  // this is to preserve locality in case of recursion

  struct level1t:public renaming_levelt
  {
  public:
    struct name_rec_hash;
    class name_record {
    public:
      name_record(const symbol2t &sym) : base_name(sym.thename) { }

      name_record(const irep_idt &name) : base_name(name) { }

      int compare(const name_record &ref) const
      {
        if (base_name.get_no() < ref.base_name.get_no())
          return -1;
        else if (base_name.get_no() > ref.base_name.get_no())
          return 1;

        return 0;
      }

      bool operator<(const name_record &ref) const
      {
        if (compare(ref) == -1)
          return true;
        return false;
      }

      bool operator==(const name_record &ref) const
      {
        if (compare(ref) == 0)
          return true;
        return false;
      }

      irep_idt base_name;

      friend struct renaming::level1t::name_rec_hash;
    };

    struct name_rec_hash
    {
      size_t operator()(const name_record &ref) const
      {
        return ref.base_name.get_no();
      }

      bool operator()(const name_record &ref, const name_record &ref2) const
      {
        return ref < ref2;
      }
    };

    typedef hash_map_cont<name_record, unsigned, name_rec_hash> current_namest;
    current_namest current_names;
    unsigned int thread_id;

    virtual void rename(expr2tc &expr);
    virtual void get_ident_name(expr2tc &symbol) const;
    virtual void remove(const expr2tc &symbol)
    {
      current_names.erase(name_record(to_symbol2t(symbol)));
    }

    void rename(const expr2tc &symbol, unsigned frame)
    {
      // Given that this is level1, use base symbol.
      unsigned &frameno = current_names[name_record(to_symbol2t(symbol))];
      assert(frameno <= frame);
      frameno = frame;
    }

    virtual void get_original_name(expr2tc &expr) const
    {
      renaming_levelt::get_original_name(expr, symbol2t::level0);
    }

    unsigned int current_number(const irep_idt &name) const;

    level1t() {}
    virtual ~level1t() { }

    virtual void print(std::ostream &out) const;
  };

  // level 2 -- SSA

  struct level2t:public renaming_levelt
  {
  protected:
    virtual void coveredinbees(expr2tc &lhs_sym, unsigned count, unsigned node_id);
  public:
    class name_record {
    public:
      name_record(const symbol2t &sym)
        : base_name(sym.thename), lev(sym.rlevel), l1_num(sym.level1_num),
          t_num(sym.thread_num)
      {
        size_t seed = 0;
        boost::hash_combine(seed, base_name.get_no());
        boost::hash_combine(seed, (uint8_t)lev);
        boost::hash_combine(seed, l1_num);
        boost::hash_combine(seed, t_num);
        hash = seed;
      }

      int compare(const name_record &ref) const
      {
        if (hash < ref.hash)
          return -1;
        else if (hash > ref.hash)
          return 1;

        if (base_name < ref.base_name)
          return -1;
        else if (ref.base_name < base_name)
          return 1;

        if (lev < ref.lev)
          return -1;
        else if (lev > ref.lev)
          return 1;

        if (l1_num < ref.l1_num)
          return -1;
        else if (l1_num > ref.l1_num)
          return 1;

        if (t_num < ref.t_num)
          return -1;
        else if (t_num > ref.t_num)
          return 1;

        return 0;
      }

      bool operator<(const name_record &ref) const
      {
        if (compare(ref) == -1)
          return true;
        return false;
      }

      bool operator==(const name_record &ref) const
      {
        if (compare(ref) == 0)
          return true;
        return false;
      }

      irep_idt base_name;
      symbol2t::renaming_level lev;
      unsigned int l1_num;
      unsigned int t_num;

      // Not a part of comparisons etc,
      size_t hash;
    };

    struct name_rec_hash
    {
      size_t operator()(const name_record &ref) const
      {
        return ref.hash;
      }

      bool operator()(const name_record &ref, const name_record &ref2) const
      { 
        return ref < ref2;
      }
    };

  public:
    virtual void make_assignment(expr2tc &lhs_symbol,
                                 const expr2tc &constant_value,
                                 const expr2tc &assigned_value);

    virtual void rename(expr2tc &expr);
    virtual void rename(expr2tc &expr, unsigned count)=0;

    virtual void get_ident_name(expr2tc &symbol) const;

    virtual void remove(const expr2tc &symbol)
    {
        current_names.erase(name_record(to_symbol2t(symbol)));
    }

    void remove(const name_record &rec)
    {
      current_names.erase(rec);
    }

    virtual void get_original_name(expr2tc &expr) const
    {
      renaming_levelt::get_original_name(expr, symbol2t::level1);
    }

    struct valuet
    {
      unsigned count;
      expr2tc constant;
      unsigned node_id;
      valuet():
        count(0),
        constant(),
        node_id(0)
      {
      }
    };

    void get_variables(std::set<name_record> &vars) const
    {
      for(current_namest::const_iterator it=current_names.begin();
          it!=current_names.end();
          it++)
      {
        vars.insert(it->first);
      }
    }

    unsigned current_number(const expr2tc &sym) const;
    unsigned current_number(const name_record &rec) const;

    level2t() { };
    virtual ~level2t() { };
    virtual level2t *clone(void) const = 0;

    virtual void print(std::ostream &out) const;
    virtual void dump() const;

  protected:
    typedef std::map<const name_record, valuet, name_rec_hash> current_namest;
    current_namest current_names;
    typedef std::map<const expr2tc, crypto_hash> current_state_hashest;
    current_state_hashest current_hashes;
  };

}

#endif /* _GOTO_SYMEX_RENAMING_H_ */
