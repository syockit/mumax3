package main

import (
	"flag"
	"fmt"
	"log"
	"strconv"
	"strings"
	"time"
)

var (
	flag_addr    = flag.String("addr", ":35360", "Serve at this network address")
	flag_scan    = flag.String("scan", "127.0.0-1.1,192.168.0.1-253", "Scan these IP address for other servers")
	flag_ports   = flag.String("ports", "35360-35361", "Scan these ports for other servers")
	flag_timeout = flag.Duration("timeout", 1*time.Second, "Portscan timeout")
	flag_http    = flag.String("http", ":8080", "Port to serve status over HTTP")
)

const MaxIPs = 1024

func main() {
	log.SetFlags(0)
	log.SetPrefix("mumax3-server: ")
	flag.Parse()

	IPs := parseIPs()
	minPort, maxPort := parsePorts()

	jobs := flag.Args()
	MainSlave(*flag_addr, IPs, minPort, maxPort, jobs)
}

func Fatal(err error) {
	if err != nil {
		log.Fatal(err)
	}
}

// Parse port range flag. E.g.:
// 	1234-1237 -> 1234, 1237
func parsePorts() (minPort, maxPort int) {
	p := *flag_ports
	split := strings.Split(p, "-")
	if len(split) > 2 {
		log.Fatal("invalid port range:", p)
	}
	minPort, _ = strconv.Atoi(split[0])
	if len(split) > 1 {
		maxPort, _ = strconv.Atoi(split[1])
	}
	if maxPort == 0 {
		maxPort = minPort
	}
	if minPort == 0 || maxPort == 0 || maxPort < minPort {
		log.Fatal("invalid port range:", p)
	}
	return
}

// init IPs from flag
func parseIPs() []string {
	var IPs []string
	defer func() {
		if err := recover(); err != nil {
			log.Fatal("invalid IP range:", *flag_scan)
		}
	}()

	p := *flag_scan
	split := strings.Split(p, ",")
	for _, s := range split {
		split := strings.Split(s, ".")
		if len(split) != 4 {
			log.Fatal("invalid IP address range:", s)
		}
		var start, stop [4]byte
		for i, s := range split {
			split := strings.Split(s, "-")
			first := atobyte(split[0])
			start[i], stop[i] = first, first
			if len(split) > 1 {
				stop[i] = atobyte(split[1])
			}
		}

		for A := start[0]; A <= stop[0]; A++ {
			for B := start[1]; B <= stop[1]; B++ {
				for C := start[2]; C <= stop[2]; C++ {
					for D := start[3]; D <= stop[3]; D++ {
						if len(IPs) > MaxIPs {
							log.Fatal("too many IP addresses to scan in", p)
						}
						IPs = append(IPs, fmt.Sprintf("%v.%v.%v.%v", A, B, C, D))
					}
				}
			}
		}
	}
	return IPs
}

func atobyte(a string) byte {
	i, err := strconv.Atoi(a)
	if err != nil {
		panic(err)
	}
	if int(byte(i)) != i {
		panic("too large")
	}
	return byte(i)
}