import csv
import json 

# Dictionaries for chapters and connectors
chapters = {
    "EventListeners": {
        "Sams.EventListener.ActiveMQ.Consumer": {},
        "Sams.EventListener.AQMessaging.Subscriber": {},
        "Sams.EventListener.AMQP.Receiver": {},
        "Sams.EventListener.File.FileSystemWatcher": {},
        "Sams.EventListener.HTTP.WebServer": {},
        "Sams.EventListener.Irc.Receive": {},
        "Sams.EventListener.MQTT.Broker": {},
        "Sams.EventListener.MQTT.Subscriber": {},
        "Sams.EventListener.OPC.Subscriber(Customer Connector)": {},
        "Sams.EventListener.SAP.RfcService": {},
        "Sams.EventListener.Scheduler.QuartzScheduler": {},
        "Sams.EventListener.Soap.SecureWebServer": {},
        "Sams.EventListener.Soap.WebServer": {},
        "Sams.EventListener.Wcf.RestService": {},
        "Sams.EventListener.Wcf.SimpleService": {}
    },
    "Sources": {
        "Sams.Connector.Database.Source.Select": {},
        "Sams.Connector.Database.Source.SelectDb2": {},
        "Sams.Connector.Database.Source.SelectPG": {},
        "Sams.Connector.eBR.Source.XsiDatabase": {},
        "Sams.Connector.eBR.Source.XsiPGDatabase": {},
        "Sams.Connector.File.Source.Binary": {},
        "Sams.Connector.File.Source.FileProperties": {},
        "Sams.Connector.File.Source.Csv": {},
        "Sams.Connector.File.Source.Json": {},
        "Sams.Connector.File.Source.Text": {},
        "Sams.Connector.File.Source.Xml": {},
        "Sams.Connector.Odata.Source.GetPayload": {},
        "Sams.Connector.OPC.Source.Read(Customer Connector)": {},
        "Sams.Connector.SAP.Source.CnxTest": {},
        "Sams.Connector.SAP.Source.Rfc": {},
        "Sams.Connector.SAP.Source.Server": {},
        "Sams.Connector.Sharepoint.Source.LibraryDocument": {},
        "Sams.Connector.Sharepoint.Source.LibraryDocumentProperties": {},
        "Sams.Connector.Soap.Source.Http": {},
        "Sams.Connector.Void.Source.Nop": {},
        "Sams.Connector.Windows.Source.Wmi": {}
    },
    "Processors": {
        "Sams.Processor.Database.Execute": {},
        "Sams.Processor.Database.Select": {},
        "Sams.Processor.File.Filter": {},
        "Sams.Processor.OPC.ExecuteMethod(Customer Connector)": {},
        "Sams.Processor.OPC.Read(Customer Connector)": {},
        "Sams.Processor.OPC.Write(Customer Connector)": {},
        "Sams.Processor.Oracle.Execute": {},
        "Sams.Processor.Oracle.Select": {},
        "Sams.Processor.Xml.AddMetadata": {},
        "Sams.Processor.Xml.Merge": {},
        "Sams.Processor.Xml.XPath": {},
        "Sams.Processor.Xml.Xsd": {},
        "Sams.Processor.Xml.Dump": {},
        "Sams.Processor.Xml.Xsl": {}
    },
    "Destinations": {
        "Sams.Connector.ActiveMQ.Destination.Publisher": {},
        "Sams.Connector.AMQP.Destination.Sender": {},
        "Sams.Connector.Database.Destination.Action": {},
        "Sams.Connector.Database.Destination.Enqueue": {},
        "Sams.Connector.Database.Destination.Execute": {},
        "Sams.Connector.eBR.Destination.Do": {},
        "Sams.Connector.eBR.Destination.DoPG": {},
        "Sams.Connector.eBR.Destination.eBRDoc": {},
        "Sams.Connector.eBR.Destination.XsiDatabase": {},
        "Sams.Connector.eBR.Destination.XsiPGDatabase": {},
        "Sams.Connector.Email.Destination.SMTP": {},
        "Sams.Connector.File.Destination.Json": {},
        "Sams.Connector.File.Destination.MergePdf": {},
        "Sams.Connector.File.Destination.Move": {},
        "Sams.Connector.File.Destination.Binary": {},
        "Sams.Connector.File.Destination.Any": {},
        "Sams.Connector.File.Destination.Pdf": {},
        "Sams.Connector.File.Destination.Text": {},
        "Sams.Connector.HTTP.Destination.Publisher": {},
        "Sams.Connector.Irc.Destination.Send": {},
        "Sams.Connector.MQTT.Destination.Publisher": {},
        "Sams.Connector.Odata.Destination.PostPayload": {},
        "Sams.Connector.OPC.Destination.Write(Customer Connector)": {},
        "Sams.Connector.Preactor.Destination.PCO": {},
        "Sams.Connector.Reporting.Destination.Docx": {},
        "Sams.Connector.Reporting.Destination.Rdlc": {},
        "Sams.Connector.SAP.Destination.RfcIdoc": {},
        "Sams.Connector.SAP.Destination.Rfc": {},
        "Sams.Connector.Soap.Destination.Http": {},
        "Sams.Connector.Void.Destination.StopProcess": {},
        "Sams.Connector.Void.Destination.Nop": {},
        "Sams.Connector.Wcf.Destination.Http": {},
        "Sams.Connector.Windows.Destination.StartProcess": {}
    }
}

def process_attributes(csv_reader, attributes_row):
    attributes = {}

    # Process the first row with attributes 
    identifier = attributes_row[2]
    description = attributes_row[3]
    value = attributes_row[4]
    attributes[identifier] = {
        'Description': description,
        'Value': value
    }

    for attr_row in csv_reader:
        if attr_row[0] == 'Configuration' or "<!--Code snippet to insert in a task-->" in attr_row:
            # Stop processing when "Configuration" is encountered
            break

        # Extracting the identifier, description, and value
        identifier = attr_row[2]
        description = attr_row[3]
        value = attr_row[4]

        # Adding the attribute to the dictionary
        attributes[identifier] = {
            'Description': description,
            'Value': value
        }
    
    # Keep track of where we are in iteration
    last_row = attr_row
    
    return attributes, last_row

def process_configuration(csv_reader, config_row):
    configuration = {}

    # Process the first row with configuration 
    identifier = config_row[2]
    description = config_row[3]
    value = config_row[4]
    configuration[identifier] = {
        'Description': description,
        'Value': value
    }

    for config_item_row in csv_reader:
        if "<!--Code snippet to insert in a task-->" in config_item_row:
            # Stop processing when the break condition is encountered
            break

        # Extracting the identifier, description, and value
        identifier = config_item_row[2]
        description = config_item_row[3]
        value = config_item_row[4]

        # Adding the configuration item to the dictionary
        configuration[identifier] = {
            'Description': description,
            'Value': value
        }

    return configuration


def update_structure_from_csv(chapters, csv_filename):
    current_chapter = None
    current_connector = None
    processed_connectors = set() 
    
    with open(csv_filename, 'r', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        
        for row in csv_reader:
            if row:
                if row[0].startswith("6- EventListeners"):
                    current_chapter = "EventListeners"
                elif row[0].startswith("7- Sources"):
                    current_chapter = "Sources"
                elif row[0].startswith("8- Processors"):
                    current_chapter = "Processors"
                elif row[0].startswith("9- Destinations"):
                    current_chapter = "Destinations"
                elif row[0] in chapters[current_chapter] and row[0] not in processed_connectors:
                    current_connector = row[0]
                    # The connectors name appears twice, also in code snippet. To avoid loops, add the connector to the set of processed connectors
                    processed_connectors.add(current_connector) 
                    # Get next row
                    next_row = next(csv_reader)
                    chapters[current_chapter][current_connector]['Description'] = ' '.join(next_row)
                    
                    # Check if the next row starts with "Assembly informations"
                    assembly_info_row = next(csv_reader)
                    if assembly_info_row and assembly_info_row[0] == "Assembly informations":
                        # Assembly information is in the same row
                        chapters[current_chapter][current_connector]['Assembly Informations'] = ' '.join(assembly_info_row[1:])
                        
                    # Check if the next row starts with "Properties"
                    properties_row = next(csv_reader)
                    if properties_row and properties_row[0] == "Properties":
                        properties = {}
                        
                        # EntityType 
                        entity_type_value = properties_row[3] 
                        properties['EntityType'] = entity_type_value

                        # BaseType
                        base_type_row = next(csv_reader)
                        base_type_value = base_type_row[3] if len(base_type_row) > 2 else ""
                        properties['BaseType'] = base_type_value

                        # ProcessorArchitecture
                        processor_arch_row = next(csv_reader)
                        processor_arch_value = processor_arch_row[3] if len(processor_arch_row) > 2 else ""
                        properties['ProcessorArchitecture'] = processor_arch_value

                        # EmbeddedSchema
                        embedded_schema_row = next(csv_reader)
                        embedded_schema_value = embedded_schema_row[3] if len(embedded_schema_row) > 2 else ""
                        properties['EmbeddedSchema'] = embedded_schema_value
                        
                        # Assign the properties dictionary to the current connector
                        chapters[current_chapter][current_connector]['Properties'] = properties
            

                    # Check if the next row starts with "Attributes"
                    attributes_row = next(csv_reader)
                    if attributes_row and attributes_row[0] == "Attributes":
                        # Process attributes and assign the result to the connector
                        # Process attributes and get the last row read
                        attributes, last_row = process_attributes(csv_reader, attributes_row)

                        chapters[current_chapter][current_connector]['Attributes'] = attributes

                    # Check if the next row starts with "Configuration"
                    if last_row is not None and (last_row[0] == "Configuration"):
                        # Process configuration and assign the result to the connector
                        chapters[current_chapter][current_connector]['Configuration'] = process_configuration(csv_reader, last_row)
                    
                

# Test with an example CSV file
csv_filename = 'C:/Users/z000p01m/Documents/Stage/Test/Sams.Documentation - Copy2.csv'
update_structure_from_csv(chapters, csv_filename)

# Print the updated chapters dictionary
# print(json.dumps(chapters["Sources"]["Sams.Connector.File.Source.Csv"], indent=2))
 
def flatten_structure(chapters):
    """
    Flatten the structure to a JSON format : id, text, title
    id : int
    text : name connector, description, attributes, configuration
    title : empty 
    """
    flattened_structure = []
    id = 0
    for chapter, connectors in chapters.items():
        for connector, connector_data in connectors.items():
            attributes_keys = ', '.join(connector_data.get('Attributes', {}).keys())
            configuration_keys = ', '.join(connector_data.get('Configuration', {}).keys())
            text = f"{connector}, {connector_data.get('Description', '')}, attributes: {attributes_keys}, configuration: {configuration_keys}"
            flattened_structure.append(
                {
                    'text': text,
                    'title': '',
                    '_id': id
                })
            id += 1
    return flattened_structure

flat_JSON = flatten_structure(chapters)

# Save the JSON to a file
with open('output.json', 'w') as f:
    for item in flat_JSON:
        f.write(json.dumps(item) + '\n')