import { TripletexApi } from './tripletexApi.js';

/**
 * Compound tools that handle common multi-step accounting flows.
 * Each function takes simple inputs and handles all lookups internally.
 */

// ---------------------------------------------------------------------------
// 1. Create Invoice Flow
// ---------------------------------------------------------------------------

export async function createInvoiceFlow(api: TripletexApi, input: {
  customerOrgNumber: string;
  lines: Array<{ productNumber?: string; description?: string; quantity: number; unitPrice: number; vatRate?: number }>;
  invoiceDate: string;
  send?: boolean;
}): Promise<any> {
  const today = input.invoiceDate;

  // Parallel lookups
  const [custRes, vatRes, bankRes] = await Promise.all([
    api.get('/customer', { organizationNumber: input.customerOrgNumber, fields: 'id,name' }),
    api.get('/ledger/vatType', { fields: 'id,name,number' }),
    api.get('/ledger/account', { number: 1920, fields: 'id,bankAccountNumber,version' }),
  ]);

  const customer = custRes.values?.[0];
  if (!customer) throw new Error(`Customer with org number ${input.customerOrgNumber} not found`);

  const bank = bankRes.values?.[0];

  // Set bank account if empty
  if (bank && !bank.bankAccountNumber) {
    await api.put(`/ledger/account/${bank.id}`, { bankAccountNumber: '86011117947', version: bank.version });
  }

  // Look up products if product numbers specified
  const productLookups = await Promise.all(
    input.lines
      .filter(l => l.productNumber)
      .map(l => api.get('/product', { number: l.productNumber, fields: 'id,name,number' }))
  );
  const productMap: Record<string, any> = {};
  for (const res of productLookups) {
    if (res.values?.[0]) {
      productMap[res.values[0].number] = res.values[0];
    }
  }

  // Find VAT types
  const vatTypes = vatRes.values || [];
  const vat25 = vatTypes.find((v: any) => v.number === '3'); // Outgoing 25%
  const vat15 = vatTypes.find((v: any) => v.number === '31'); // Outgoing 15% (food)
  const vat0 = vatTypes.find((v: any) => v.number === '6'); // 0% exempt

  function findVatType(rate?: number) {
    if (rate === 0) return vat0;
    if (rate === 15) return vat15;
    return vat25; // Default 25%
  }

  // Create order
  const order = (await api.post('/order', {
    customer: { id: customer.id },
    orderDate: today,
    deliveryDate: today,
  })).value;

  // Build order lines
  const orderLines = input.lines.map(line => {
    const product = line.productNumber ? productMap[line.productNumber] : null;
    const vatType = findVatType(line.vatRate);
    return {
      order: { id: order.id },
      ...(product ? { product: { id: product.id } } : {}),
      description: line.description || product?.name || '',
      count: line.quantity,
      unitPriceExcludingVatCurrency: line.unitPrice,
      ...(vatType ? { vatType: { id: vatType.id } } : {}),
    };
  });

  // Batch create order lines
  if (orderLines.length > 1) {
    await api.postList('/order/orderline/list', orderLines);
  } else if (orderLines.length === 1) {
    await api.post('/order/orderline', orderLines[0]);
  }

  // Convert to invoice
  const invoice = (await api.put(`/order/${order.id}/:invoice`, {}, { invoiceDate: today })).value;

  // Send if requested
  if (input.send) {
    await api.put(`/invoice/${invoice.id}/:send`, {}, { sendType: 'EMAIL' });
  }

  return {
    customerId: customer.id,
    customerName: customer.name,
    orderId: order.id,
    invoiceId: invoice.id,
    invoiceNumber: invoice.invoiceNumber,
    sent: !!input.send,
  };
}

// ---------------------------------------------------------------------------
// 2. Register Payment
// ---------------------------------------------------------------------------

export async function registerPayment(api: TripletexApi, input: {
  customerOrgNumber: string;
  amount?: number;
  paymentDate: string;
  paidAmountCurrency?: number;
}): Promise<any> {
  // Parallel lookups
  const [custRes, payTypeRes] = await Promise.all([
    api.get('/customer', { organizationNumber: input.customerOrgNumber, fields: 'id,name' }),
    api.get('/invoice/paymentType', { fields: 'id,description' }),
  ]);

  const customer = custRes.values?.[0];
  if (!customer) throw new Error(`Customer ${input.customerOrgNumber} not found`);

  const bankPayType = payTypeRes.values?.find((p: any) => /bank/i.test(p.description));
  if (!bankPayType) throw new Error('No bank payment type found');

  // Find unpaid invoice
  const invRes = await api.get('/invoice', {
    customerId: customer.id,
    invoiceDateFrom: '2020-01-01',
    invoiceDateTo: '2030-01-01',
    fields: 'id,invoiceNumber,amountCurrency,amountOutstanding',
  });

  const unpaid = invRes.values?.find((i: any) => i.amountOutstanding > 0);
  if (!unpaid) throw new Error(`No unpaid invoice found for customer ${customer.name}`);

  const paidAmount = input.amount || unpaid.amountCurrency;

  const params: any = {
    paymentTypeId: bankPayType.id,
    paidAmount,
    paymentDate: input.paymentDate,
  };
  if (input.paidAmountCurrency) {
    params.paidAmountCurrency = input.paidAmountCurrency;
  }

  await api.put(`/invoice/${unpaid.id}/:payment`, {}, params);

  return {
    invoiceId: unpaid.id,
    invoiceNumber: unpaid.invoiceNumber,
    paidAmount,
    customerName: customer.name,
  };
}

// ---------------------------------------------------------------------------
// 3. Create Supplier Invoice (Voucher)
// ---------------------------------------------------------------------------

export async function createSupplierInvoiceVoucher(api: TripletexApi, input: {
  supplierName: string;
  supplierOrgNumber?: string;
  invoiceNumber: string;
  grossAmount: number;
  expenseAccountNumber: number;
  vatRate?: number;
  date: string;
}): Promise<any> {
  const vatRate = input.vatRate ?? 25;
  const netAmount = Math.round((input.grossAmount / (1 + vatRate / 100)) * 100) / 100;

  // Parallel lookups
  const lookups = await Promise.all([
    input.supplierOrgNumber
      ? api.get('/supplier', { organizationNumber: input.supplierOrgNumber, fields: 'id,name' })
      : api.get('/supplier', { fields: 'id,name', count: 1000 }),
    api.get('/ledger/account', { number: input.expenseAccountNumber, fields: 'id,number,name' }),
    api.get('/ledger/account', { number: 2400, fields: 'id,number,name' }),
    api.get('/ledger/vatType', { typeOfVat: 'INCOMING', fields: 'id,name,percentage' }),
  ]);

  let supplier = input.supplierOrgNumber
    ? lookups[0].values?.[0]
    : lookups[0].values?.find((s: any) => s.name?.toLowerCase().includes(input.supplierName.toLowerCase()));

  // Create supplier if not found
  if (!supplier) {
    supplier = (await api.post('/supplier', {
      name: input.supplierName,
      ...(input.supplierOrgNumber ? { organizationNumber: input.supplierOrgNumber } : {}),
    })).value;
  }

  const expenseAccount = lookups[1].values?.[0];
  if (!expenseAccount) throw new Error(`Account ${input.expenseAccountNumber} not found`);

  const apAccount = lookups[2].values?.[0];
  if (!apAccount) throw new Error('Account 2400 not found');

  // Find incoming VAT type matching rate
  const incomingVatTypes = lookups[3].values || [];
  const vatType = incomingVatTypes.find((v: any) => Math.abs(v.percentage - vatRate) < 0.1);

  // Create voucher
  const voucher = (await api.post('/ledger/voucher', {
    date: input.date,
    description: `${input.supplierName} - ${input.invoiceNumber}`,
    vendorInvoiceNumber: input.invoiceNumber,
    postings: [
      {
        row: 1,
        account: { id: expenseAccount.id },
        amount: netAmount,
        date: input.date,
        description: expenseAccount.name,
        ...(vatType ? { vatType: { id: vatType.id } } : {}),
      },
      {
        row: 2,
        account: { id: apAccount.id },
        amount: -input.grossAmount,
        date: input.date,
        description: `${input.supplierName} - ${input.invoiceNumber}`,
        supplier: { id: supplier.id },
      },
    ],
  })).value;

  return {
    voucherId: voucher.id,
    voucherNumber: voucher.number,
    supplierId: supplier.id,
    supplierName: supplier.name || input.supplierName,
    netAmount,
    grossAmount: input.grossAmount,
    vatAmount: input.grossAmount - netAmount,
  };
}

// ---------------------------------------------------------------------------
// 4. Setup Project (with entitlements)
// ---------------------------------------------------------------------------

export async function setupProject(api: TripletexApi, input: {
  projectName: string;
  customerOrgNumber?: string;
  projectManagerEmail: string;
  startDate: string;
  budget?: number;
  isInternal?: boolean;
  createActivity?: boolean;
  activityName?: string;
}): Promise<any> {
  // Parallel lookups
  const lookups = await Promise.all([
    api.get('/employee', { email: input.projectManagerEmail, fields: 'id,firstName,lastName' }),
    api.get('/department', { fields: 'id,name', count: 1 }),
    ...(input.customerOrgNumber
      ? [api.get('/customer', { organizationNumber: input.customerOrgNumber, fields: 'id,name' })]
      : []),
  ]);

  const employee = lookups[0].values?.[0];
  if (!employee) throw new Error(`Employee ${input.projectManagerEmail} not found`);

  const dept = lookups[1].values?.[0];
  const customer = input.customerOrgNumber ? lookups[2]?.values?.[0] : null;

  // Grant entitlements first
  await api.put('/employee/entitlement/:grantEntitlementsByTemplate', {}, {
    employeeId: employee.id,
    template: 'ALL_PRIVILEGES',
  });

  // Create project
  const projectData: any = {
    name: input.projectName,
    projectManager: { id: employee.id },
    startDate: input.startDate,
    ...(dept ? { department: { id: dept.id } } : {}),
    ...(customer ? { customer: { id: customer.id } } : {}),
    ...(input.isInternal !== undefined ? { isInternal: input.isInternal } : {}),
    ...(input.budget ? { fixedprice: input.budget, isFixedPrice: true } : {}),
  };

  const project = (await api.post('/project', projectData)).value;

  // Create activity and link to project
  let activity = null;
  if (input.createActivity !== false) {
    // Step 1: Create the activity
    const newActivity = (await api.post('/activity', {
      name: input.activityName || input.projectName,
      activityType: 'PROJECT_GENERAL_ACTIVITY',
    })).value;
    // Step 2: Link to project
    activity = (await api.post('/project/projectActivity', {
      project: { id: project.id },
      activity: { id: newActivity.id },
    })).value;
  }

  return {
    projectId: project.id,
    projectName: project.name,
    projectManagerId: employee.id,
    customerId: customer?.id,
    activityId: activity?.id,
  };
}
