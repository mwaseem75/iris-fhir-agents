Class fhirsetup.Setup
{

ClassMethod SetupFHIRServer(ns = "FHIRSERVER", appKey = "/fhir/r4", testLoad = 1, interactionsStrategy As %String(VALUELIST=",Json,JsonAdvSql") = "Json") As %Status
{
    set st=$$$OK
    Write !,"=== Initialization with parameters: Namespace: "_ns_", Webapp: "_appKey_", AddTestData: "_testLoad,!
    zn "HSLIB"
    set namespace=ns
    ;Set appKey = "/fhir/r4"
    if interactionsStrategy = "Json" {
        set strategyClass = "HS.FHIRServer.Storage.Json.InteractionsStrategy"
    } elseif interactionsStrategy = "JsonAdvSql" {
        set strategyClass = "HS.FHIRServer.Storage.JsonAdvSQL.InteractionsStrategy"
    }
    set metadataPackages = $lb("hl7.fhir.r4.core@4.0.1")
    Set metadataConfigKey = "HL7v40"
    ;set importdir="/opt/irisapp/src"

    //Install a Foundation namespace and change to it
    //Do ##class(HS.HC.Util.Installer).InstallFoundation(namespace)
    Do ##class(HS.Util.Installer.Foundation).Install(namespace)

    zn namespace

    // Install elements that are required for a FHIR-enabled namespace
    do ##class(HS.FHIRServer.Installer).InstallNamespace()

    // Install an instance of a FHIR Service into the current namespace
    if '##class(HS.FHIRServer.ServiceAdmin).EndpointExists(appKey) {
       do ##class(HS.FHIRServer.Installer).InstallInstance(appKey, strategyClass, metadataPackages)
    }

    set strategy = ##class(HS.FHIRServer.API.InteractionsStrategy).GetStrategyForEndpoint(appKey)
    set config = strategy.GetServiceConfigData()
    set config.DebugMode = 4
    do strategy.SaveServiceConfigData(config)
    // Create RAG Vector Search table for clinical guidelines
    Write !,"=== Creating RAG Vector Search table..."
    zn namespace
    Set createSQL = "CREATE TABLE RAG.ClinicalGuidelines ("_
                "id VARCHAR(100) PRIMARY KEY, "_
                "source VARCHAR(200), "_
                "topic VARCHAR(200), "_
                "content VARCHAR(8000), "_
                "embedding VECTOR(DOUBLE, 1536))"
    set stmt = ##class(%SQL.Statement).%New()
    set sc = stmt.%Prepare(createSQL)
    if $$$ISOK(sc) {
        set rs = stmt.%Execute()
        Write !,"=== RAG.ClinicalGuidelines table ready"
    } else {
        Write !,"=== RAG table note: ",$System.Status.GetErrorText(sc)
    }
    if testLoad {
      set dir="/home/irisowner/irisdev/data/fhir/"
      write !," ====== Load test data in "_ns_" from "_dir
      set st=##class(HS.FHIRServer.Tools.DataLoader).SubmitResourceFiles(dir, "FHIRServer", appKey)
      if 'st write $System.Status.GetErrorText(st)
    }
    quit st
}

ClassMethod LoadPatientData(path, namespace, appKey) As %Status
{
    quit ##class(HS.FHIRServer.Tools.DataLoader).SubmitResourceFiles(path, namespace, appKey)
}

}