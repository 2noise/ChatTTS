package main

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"os"
)

func main() {
	var buf [32]byte
	h := sha256.New()
	lst := make([]any, 0, 64)
	for _, fname := range files {
		f, err := os.Open(fname)
		if err != nil {
			panic(err)
		}
		_, err = io.Copy(h, f)
		if err != nil {
			panic(err)
		}
		s := hex.EncodeToString(h.Sum(buf[:0]))
		fmt.Println("sha256 of", fname, "=", s)
		lst = append(lst, s)
		h.Reset()
		f.Close()
	}
	f, err := os.Create("ChatTTS/res/sha256_map.json")
	if err != nil {
		panic(err)
	}
	_, err = fmt.Fprintf(f, jsontmpl, lst...)
	if err != nil {
		panic(err)
	}
}
