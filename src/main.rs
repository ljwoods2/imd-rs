use std::io::{self, Read, Write};
use std::net::{TcpListener, TcpStream};

#[repr(i32)]
#[derive(Debug)]
enum IMDMessageType {
    Disconnect,
    Energies,
    FCoords,
    Go,
    Handshake,
    Kill,
    MDComm,
    Pause,
    TRate,
    IOError,
    SessionInfo,
    Resume,
    Time,
    Box,
    Velocities,
    Forces,
    Wait,
}

impl std::convert::TryFrom<i32> for IMDMessageType {
    type Error = &'static str;

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(IMDMessageType::Disconnect),
            1 => Ok(IMDMessageType::Energies),
            2 => Ok(IMDMessageType::FCoords),
            3 => Ok(IMDMessageType::Handshake),
            _ => Err("Invalid value for IMDMessageType"),
        }
    }
}

fn main() {
    println!("Hello, world!");
}

#[derive(Debug)]
struct IMDHeader {
    header_type: IMDMessageType,
    length: i32,
}

fn expect_imd_handshake(
    stream: &mut TcpStream,
) -> io::Result<i32> {

    let mut buf = [0; 8];

    match stream.read_exact(&mut buf) {
        Ok(()) => {
            let handshake_header_int = i32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]]);

            
            let length = i32::from_be_bytes([buf[4], buf[5], buf[6], buf[7]]);
        }
        Err(e) => {
            eprintln!("Error reading from stream: {}", e);
        }
    }
    1;
}

fn expect_imd_header(
    stream: &mut TcpStream,
    imd_type: IMDMessageType,
    buf: &mut [u8; 8],
    val: Option<i32>,
) -> io::Result<IMDHeader> {

    match stream.read_exact(buf) {
        Ok(()) => {
            let header_type = i32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]]);

            let length = i32::from_be_bytes([buf[4], buf[5], buf[6], buf[7]]);
        }
        Err(e) => {
            eprintln!("Error reading from stream: {}", e);
        }
    }

    if let Some(expected_val) = val {
        if 
    }

    Ok()
}
