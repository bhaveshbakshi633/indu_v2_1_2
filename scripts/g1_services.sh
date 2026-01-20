#!/bin/bash
# G1 Remote Services Manager
# Saari external services jo health checkup me hain
# Usage: ./g1_services.sh [start|stop|status] [service|all]

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# Service Configs: host|password|start_cmd|stop_cmd|port|name
declare -A SERVICES
SERVICES=(
    ["chatterbox"]="ssi@172.16.6.19|Ssi@113322|cd ~/Music/chatterbox && source .venv/bin/activate && nohup python tts_server.py > /tmp/chatterbox.log 2>&1 &|pkill -f tts_server.py|8000|Chatterbox TTS"
    ["whisper"]="b@172.16.4.250|1997|cd ~/Whisper && source whisper_venv/bin/activate && nohup python3 whisper_server.py > /tmp/whisper.log 2>&1 &|pkill -f whisper_server.py|8001|Whisper STT"
    ["ollama"]="isaac@172.16.4.226|7410|sudo systemctl start ollama|sudo systemctl stop ollama; pkill -f 'ollama serve'|11434|Ollama LLM"
    ["brain"]="ssi@172.16.6.19|Ssi@113322|source /opt/ros/humble/setup.bash && cd ~/Downloads/indu_brain_v2_1_1 && nohup python3 server.py > /tmp/brain_v2.log 2>&1 &|pkill -9 -f '[p]ython3 server.py'|8080|Voice Assistant"
)

# Service order for start (dependencies first) and stop (reverse)
START_ORDER="chatterbox whisper ollama brain"
STOP_ORDER="brain ollama whisper chatterbox"

get_field() {
    echo "${SERVICES[$1]}" | cut -d'|' -f$2
}

ssh_exec() {
    local host=$1 pass=$2 cmd=$3
    # sudo commands ke liye password pipe karo
    if [[ "$cmd" == *"sudo"* ]]; then
        # Replace 'sudo' with 'echo pass | sudo -S'
        local sudo_cmd="${cmd//sudo/echo $pass | sudo -S}"
        sshpass -p "$pass" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "$host" "$sudo_cmd" 2>/dev/null
    else
        sshpass -p "$pass" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 "$host" "$cmd" 2>/dev/null
    fi
}

check_health() {
    local svc=$1
    local ip=$(get_field "$svc" 1 | cut -d'@' -f2)
    local port=$(get_field "$svc" 5)
    local name=$(get_field "$svc" 6)

    # Service-specific endpoints aur protocols
    local endpoint="/health"
    local protocol="http"

    [[ "$svc" == "ollama" ]] && endpoint="/api/tags"
    # brain HTTPS pe chalta hai aur /api/health endpoint hai
    [[ "$svc" == "brain" ]] && endpoint="/api/health" && protocol="https"

    if curl -sk --connect-timeout 3 "${protocol}://$ip:$port$endpoint" > /dev/null 2>&1; then
        echo -e "  ${GREEN}●${NC} $name ${CYAN}($ip:$port)${NC} - ${GREEN}Running${NC}"
        return 0
    else
        echo -e "  ${RED}○${NC} $name ${CYAN}($ip:$port)${NC} - ${RED}Down${NC}"
        return 1
    fi
}

stop_service() {
    local svc=$1
    local host=$(get_field "$svc" 1)
    local pass=$(get_field "$svc" 2)
    local stop_cmd=$(get_field "$svc" 4)
    local name=$(get_field "$svc" 6)

    echo -e "  ${YELLOW}Stopping${NC} $name..."

    if [[ "$host" == "ssi@172.16.6.19" ]]; then
        eval "$stop_cmd" 2>/dev/null
    else
        ssh_exec "$host" "$pass" "$stop_cmd"
    fi

    sleep 1
    echo -e "  ${GREEN}✓${NC} $name stopped"
}

start_service() {
    local svc=$1
    local host=$(get_field "$svc" 1)
    local pass=$(get_field "$svc" 2)
    local start_cmd=$(get_field "$svc" 3)
    local name=$(get_field "$svc" 6)

    echo -e "  ${YELLOW}Starting${NC} $name..."

    if [[ "$host" == "ssi@172.16.6.19" ]]; then
        eval "$start_cmd"
    else
        ssh_exec "$host" "$pass" "$start_cmd"
    fi

    sleep 3
    check_health "$svc"
}

do_action() {
    local action=$1
    local target=$2

    echo ""

    if [[ "$target" == "all" ]]; then
        case $action in
            status)
                echo -e "${CYAN}═══ G1 Services Status ═══${NC}"
                for svc in $START_ORDER; do check_health "$svc"; done
                ;;
            stop)
                echo -e "${RED}═══ Stopping All Services ═══${NC}"
                for svc in $STOP_ORDER; do stop_service "$svc"; done
                echo -e "\n${GREEN}All services stopped${NC}"
                ;;
            start)
                echo -e "${GREEN}═══ Starting All Services ═══${NC}"
                for svc in $START_ORDER; do start_service "$svc"; done
                echo -e "\n${GREEN}All services started${NC}"
                ;;
            restart)
                echo -e "${YELLOW}═══ Restarting All Services ═══${NC}"
                for svc in $STOP_ORDER; do stop_service "$svc"; done
                echo ""
                for svc in $START_ORDER; do start_service "$svc"; done
                ;;
        esac
    else
        if [[ -z "${SERVICES[$target]}" ]]; then
            echo -e "${RED}Error:${NC} Unknown service '$target'"
            echo "Available: chatterbox, whisper, ollama, brain, all"
            exit 1
        fi
        case $action in
            status) check_health "$target" ;;
            stop) stop_service "$target" ;;
            start) start_service "$target" ;;
            restart) stop_service "$target"; sleep 1; start_service "$target" ;;
        esac
    fi
    echo ""
}

show_help() {
    echo ""
    echo -e "${CYAN}G1 Remote Services Manager${NC}"
    echo ""
    echo "Usage: $0 [action] [service]"
    echo ""
    echo "Actions:"
    echo "  status   - Check service health"
    echo "  start    - Start service(s)"
    echo "  stop     - Stop service(s)"
    echo "  restart  - Restart service(s)"
    echo ""
    echo "Services (from G1 health checkup):"
    echo "  chatterbox - Chatterbox TTS    (172.16.6.19:8000)"
    echo "  whisper    - Whisper STT       (172.16.4.250:8001)"
    echo "  ollama     - Ollama LLM        (172.16.4.226:11434)"
    echo "  brain      - Voice Assistant   (172.16.6.19:8080)"
    echo "  all        - All services"
    echo ""
    echo "Examples:"
    echo "  $0 status all      # Saari services ka status"
    echo "  $0 stop all        # Saari services band karo"
    echo "  $0 stop whisper    # Sirf whisper band karo"
    echo "  $0 restart brain   # Voice assistant restart"
    echo ""
}

# Main
[[ $# -lt 1 ]] && { show_help; exit 0; }

case $1 in
    status|start|stop|restart) do_action "$1" "${2:-all}" ;;
    -h|--help|help) show_help ;;
    *) echo -e "${RED}Unknown action:${NC} $1"; show_help; exit 1 ;;
esac
