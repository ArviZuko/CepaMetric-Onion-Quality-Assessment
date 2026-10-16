; Onion AI Inspection System
; Compile with Inno Setup 6: ISCC OnionAIInspection.iss

#define MyAppName "Onion AI Inspection System"
#define MyAppVersion "2.1"
#define MyAppPublisher "SMART LAB"
#define MyAppExeName "OnionAIInspection.vbs"

[Setup]
AppId={{A7C3E1B4-6D28-4F0A-9B55-1E8C4D2A7F60}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={localappdata}\Programs\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir=..\Installer
OutputBaseFilename=installer
; The installer.exe already in Installer\ is the 8 October 2026 build.
; This script was not compiled again. It still packages E:\OnionPack.
SetupIconFile=..\assets\app.ico
UninstallDisplayIcon={app}\SRC\app.ico
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
CloseApplications=no

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Files]
Source: "E:\OnionPack\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{sys}\wscript.exe"; Parameters: """{app}\{#MyAppExeName}"""; WorkingDir: "{app}"; IconFilename: "{app}\SRC\app.ico"; Comment: "Start the onion inspection software"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{sys}\wscript.exe"; Parameters: """{app}\{#MyAppExeName}"""; WorkingDir: "{app}"; IconFilename: "{app}\SRC\app.ico"; Comment: "Start the onion inspection software"

[Run]
Filename: "{sys}\wscript.exe"; Parameters: """{app}\{#MyAppExeName}"""; Description: "Launch {#MyAppName}"; Flags: postinstall nowait skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}"
