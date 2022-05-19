#! /bin/sh


cat $1 | sed 's/BLEU/ LEU/g' | sed 's/BSER/ SER/g'  | sed 's/BALA/ ALA/g'  | sed 's/BASN/ ASN/g'  |\
    sed 's/BLYS/ LYS/g' | sed 's/BGLU/ GLU/g'  | sed 's/BPRO/ PRO/g' | sed 's/BILE/ ILE/g'|\
    sed 's/BCYS/ CYS/g' | sed 's/BTYR/ TYR/g' | sed 's/BTHR/ THR/g' | sed 's/BASP/ ASP/g' |\
    sed 's/BMET/ MET/g' | sed 's/BGLN/ GLN/g' | sed 's/BGLY/ GLY/g' | sed 's/BTRP/ TRP/g' |\
    sed 's/BARG/ ARG/g' | sed 's/BPHE/ PHE/g' | sed 's/BHIS/ HIS/g' | sed 's/BVAL/ VAL/g'> tmp.pdb

cat tmp.pdb | grep -vE "[AC]HIS|[AC]PHE|[AC]ALA|[AC]LEU|[AC]SER|[AC]ASN|[AC]LYS|[AC]GLU|[AC]PRO|[AC]ILE|[AC]CYS|[AC]TYR|[AC]THR|[AC]ASP|[AC]MET|[AC]GLN|[AC]GLY|[AC]TRP|[AC]ARG|[AC]VAL" > clean_$1
rm tmp.pdb
