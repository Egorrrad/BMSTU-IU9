package main

import (
	"flag"
	"fmt"
	"log"
)

var (
	PrintMaxLen = 80
	Debug       = false
)

func main() {
	match := flag.Int("match", 0, "match score")
	mismatch := flag.Int("mismatch", 0, "mismatch score")
	gap := flag.Int("gap", -10, "gap penalty")
	debug := flag.Bool("debug", false, "debug mode")

	flag.Parse()

	args := flag.Args()
	if len(args) < 2 {
		log.Fatal("requires seq1 and seq2 arguments")
	}
	seq1 := args[0]
	seq2 := args[1]

	Debug = *debug
	fmt.Println(*match, *mismatch, *gap)

	var score int
	var aln1, aln2 string

	if *match != 0 && *mismatch != 0 {
		scoreFun := func(x, y uint8, matchScore, mismatchScore int) int {
			if x == y {
				return *match
			}
			return *mismatch
		}

		score, aln1, aln2 = NeedlemanWunsch(seq1, seq2, scoreFun, *gap)
	} else {
		if *match != 0 || *mismatch != 0 {
			log.Fatal("match and mismatch must be specified together")
		}

		score, aln1, aln2 = NeedlemanWunsch(seq1, seq2, scoreFun, *gap)
	}

	printResults(aln1, aln2, score, nil)
}
