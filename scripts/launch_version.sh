#!/bin/bash
# G1 Version Selector Script
# v2_1, v2_tanay, v3_voice ke beech switch karne ke liye

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo ""
echo -e "${CYAN}═══════════════════════════════════════${NC}"
echo -e "${CYAN}       G1 Robot Version Selector       ${NC}"
echo -e "${CYAN}═══════════════════════════════════════${NC}"
echo ""
echo -e "  ${GREEN}1${NC}) v2_1      - Original stable version"
echo -e "  ${GREEN}2${NC}) v2_tanay  - Current production (SAFE)"
echo -e "  ${GREEN}3${NC}) v3_voice  - Voice-driven experimental"
echo ""
echo -e "  ${YELLOW}0${NC}) Stop all running versions"
echo ""
read -p "Select version to launch [1-3, 0 to stop]: " choice

DEPLOYED_DIR="/home/unitree/deployed"

stop_all() {
    echo -e "${YELLOW}Stopping all running versions...${NC}"
    cd $DEPLOYED_DIR/v2_1 2>/dev/null && docker compose down 2>/dev/null
    cd $DEPLOYED_DIR/v2_tanay 2>/dev/null && docker compose down 2>/dev/null
    cd $DEPLOYED_DIR/v3_voice 2>/dev/null && docker compose down 2>/dev/null
    pkill -f g1_orchestrator 2>/dev/null
    pkill -f arm_controller 2>/dev/null
    pkill -f action_gatekeeper 2>/dev/null
    echo -e "${GREEN}All versions stopped${NC}"
}

case $choice in
    0)
        stop_all
        ;;
    1)
        stop_all
        echo -e "${GREEN}Launching v2_1...${NC}"
        cd $DEPLOYED_DIR/v2_1
        docker compose up -d
        echo -e "${GREEN}v2_1 launched${NC}"
        ;;
    2)
        stop_all
        echo -e "${GREEN}Launching v2_tanay (production)...${NC}"
        cd $DEPLOYED_DIR/v2_tanay
        docker compose up -d
        echo -e "${GREEN}v2_tanay launched${NC}"
        ;;
    3)
        stop_all
        echo -e "${YELLOW}Launching v3_voice (experimental)...${NC}"
        cd $DEPLOYED_DIR/v3_voice
        docker compose up -d
        echo -e "${GREEN}v3_voice launched${NC}"
        ;;
    *)
        echo -e "${RED}Invalid choice${NC}"
        exit 1
        ;;
esac

echo ""
echo -e "${CYAN}Current status:${NC}"
docker ps --format "table {{.Names}}\t{{.Status}}"
